/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "DistributedMesh.h"
#include "../IO/MeshIO.h"
#include <array>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <limits>
#include <cctype>
#include <cstring>
#include <type_traits>

#ifndef MESH_HAS_MPI
// When MPI is not enabled, do not compile the real DistributedMesh implementation.
// Provide an empty translation unit to satisfy build systems.
namespace svmp { }
#else
#include <fstream>
#include <iomanip>
#include <iterator>
#include <cmath>

#include "../Fields/MeshFields.h"
#include "../Topology/MeshTopology.h"
#include "../Topology/CellTopology.h"

#if defined(SVMP_HAS_PARMETIS)
// ParMETIS includes <metis.h> and defines idx_t/real_t.
#include <parmetis.h>
#elif defined(SVMP_HAS_METIS)
// METIS defines idx_t/real_t.
#include <metis.h>
#endif

#ifdef MESH_HAS_MPI
#include <mpi.h>

namespace {
    // Helper structures and functions for MPI operations
    bool gids_are_local_iota(const std::vector<svmp::gid_t>& gids) {
        for (size_t i = 0; i < gids.size(); ++i) {
            if (gids[i] != static_cast<svmp::gid_t>(i)) return false;
        }
        return true;
    }

    uint64_t fnv1a64_append(uint64_t h, uint64_t x) {
        constexpr uint64_t kPrime = 1099511628211ULL;
        h ^= x;
        h *= kPrime;
        return h;
    }

    svmp::gid_t stable_gid_from_sorted_vertex_gids(const std::vector<svmp::gid_t>& sorted) {
        uint64_t h = 1469598103934665603ULL;
        h = fnv1a64_append(h, static_cast<uint64_t>(sorted.size()));
        for (const auto& gid : sorted) {
            h = fnv1a64_append(h, static_cast<uint64_t>(gid));
        }
        // Ensure non-negative GIDs so gather_shared_entities treats them as valid.
        return static_cast<svmp::gid_t>(h & 0x7fffffffffffffffULL);
    }

    void ensure_canonical_face_gids(svmp::MeshBase& mesh) {
        // In serial, MeshBase assigns face GIDs as local iota. In MPI builds we need face
        // IDs that are consistent across ranks so shared-face detection and ghost exchange
        // patterns are meaningful.
        const auto& current = mesh.face_gids();
        if (current.empty()) return;
        if (!gids_are_local_iota(current)) return;

        const auto& vertex_gids = mesh.vertex_gids();
        if (vertex_gids.size() != mesh.n_vertices()) return;

        std::vector<svmp::gid_t> new_face_gids(mesh.n_faces(), svmp::INVALID_GID);
        std::unordered_map<svmp::gid_t, std::vector<svmp::gid_t>> seen;
        seen.reserve(mesh.n_faces() * 2);

        for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh.n_faces()); ++f) {
            auto [verts, nverts] = mesh.face_vertices_span(f);
            if (!verts || nverts == 0) continue;

            std::vector<svmp::gid_t> face_vertices;
            face_vertices.reserve(nverts);

            bool valid = true;
            for (size_t i = 0; i < nverts; ++i) {
                const auto v = verts[i];
                if (v < 0 || static_cast<size_t>(v) >= vertex_gids.size()) {
                    valid = false;
                    break;
                }
                const auto gid = vertex_gids[static_cast<size_t>(v)];
                if (gid == svmp::INVALID_GID) {
                    valid = false;
                    break;
                }
                face_vertices.push_back(gid);
            }

            if (!valid) continue;
            std::sort(face_vertices.begin(), face_vertices.end());
            face_vertices.erase(std::unique(face_vertices.begin(), face_vertices.end()),
                                face_vertices.end());

            const svmp::gid_t gid = stable_gid_from_sorted_vertex_gids(face_vertices);
            const auto [it, inserted] = seen.emplace(gid, face_vertices);
            if (!inserted) {
                // Either the mesh contains duplicate faces or we hit a hash collision.
                throw std::runtime_error("DistributedMesh: canonical face GID collision detected");
            }
            new_face_gids[static_cast<size_t>(f)] = gid;
        }

        // Only update if we successfully assigned all GIDs.
        for (const auto gid : new_face_gids) {
            if (gid == svmp::INVALID_GID) return;
        }

        mesh.set_face_gids(std::move(new_face_gids));
    }

    void ensure_canonical_edge_gids(svmp::MeshBase& mesh) {
        // In serial, MeshBase assigns edge GIDs as local iota. In MPI builds we need edge
        // IDs that are consistent across ranks so shared-edge detection and ghost exchange
        // patterns are meaningful.
        const auto& current = mesh.edge_gids();
        if (current.empty()) return;
        if (!gids_are_local_iota(current)) return;

        const auto& vertex_gids = mesh.vertex_gids();
        if (vertex_gids.size() != mesh.n_vertices()) return;

        std::vector<svmp::gid_t> new_edge_gids(mesh.n_edges(), svmp::INVALID_GID);
        std::unordered_map<svmp::gid_t, std::array<svmp::gid_t,2>> seen;
        seen.reserve(mesh.n_edges() * 2);

        for (svmp::index_t e = 0; e < static_cast<svmp::index_t>(mesh.n_edges()); ++e) {
            const auto ev = mesh.edge_vertices(e);
            const auto v0 = ev[0];
            const auto v1 = ev[1];
            if (v0 < 0 || v1 < 0) continue;
            if (static_cast<size_t>(v0) >= vertex_gids.size() || static_cast<size_t>(v1) >= vertex_gids.size()) {
                continue;
            }
            const auto g0 = vertex_gids[static_cast<size_t>(v0)];
            const auto g1 = vertex_gids[static_cast<size_t>(v1)];
            if (g0 == svmp::INVALID_GID || g1 == svmp::INVALID_GID) continue;

            std::vector<svmp::gid_t> verts = {g0, g1};
            std::sort(verts.begin(), verts.end());
            verts.erase(std::unique(verts.begin(), verts.end()), verts.end());

            const svmp::gid_t gid = stable_gid_from_sorted_vertex_gids(verts);
            const auto [it, inserted] = seen.emplace(gid, std::array<svmp::gid_t,2>{{verts[0], (verts.size() > 1 ? verts[1] : verts[0])}});
            if (!inserted) {
                throw std::runtime_error("DistributedMesh: canonical edge GID collision detected");
            }
            new_edge_gids[static_cast<size_t>(e)] = gid;
        }

        // Only update if we successfully assigned all GIDs.
        for (const auto gid : new_edge_gids) {
            if (gid == svmp::INVALID_GID) return;
        }

        mesh.set_edge_gids(std::move(new_edge_gids));
    }

#if defined(SVMP_HAS_METIS)
    idx_t checked_idx_cast(std::uint64_t value, const char* what) {
        const auto max_idx = static_cast<std::uint64_t>(std::numeric_limits<idx_t>::max());
        if (value > max_idx) {
            throw std::runtime_error(std::string("DistributedMesh: ") + what +
                                     " exceeds METIS idx_t range");
        }
        return static_cast<idx_t>(value);
    }

    std::vector<svmp::rank_t> partition_cells_block(size_t n_cells, int n_parts) {
        std::vector<svmp::rank_t> part(n_cells, 0);
        if (n_parts <= 0 || n_cells == 0) {
            return part;
        }

        const size_t parts = static_cast<size_t>(n_parts);
        const size_t cells_per_part = n_cells / parts;
        const size_t extra = n_cells % parts;

        size_t cursor = 0;
        for (svmp::rank_t r = 0; r < n_parts; ++r) {
            const size_t n_for_rank =
                cells_per_part + (static_cast<size_t>(r) < extra ? 1u : 0u);
            for (size_t i = 0; i < n_for_rank && cursor + i < n_cells; ++i) {
                part[cursor + i] = r;
            }
            cursor += n_for_rank;
        }
        return part;
    }

    std::vector<svmp::rank_t> partition_cells_metis(const svmp::MeshBase& mesh,
                                                    int n_parts,
                                                    svmp::PartitionHint hint) {
        const size_t n_cells = mesh.n_cells();
        if (n_parts <= 1 || n_cells == 0) {
            return std::vector<svmp::rank_t>(n_cells, 0);
        }
        if (n_cells < static_cast<size_t>(n_parts)) {
            return partition_cells_block(n_cells, n_parts);
        }

        std::vector<svmp::offset_t> cell2cell_offsets;
        std::vector<svmp::index_t> cell2cell;
        svmp::MeshTopology::build_cell2cell(mesh, cell2cell_offsets, cell2cell);

        // Degenerate graphs (no adjacency) do not benefit from METIS and can be
        // sensitive to tie-breaking; fall back to deterministic block partitioning.
        if (cell2cell.empty()) {
            return partition_cells_block(n_cells, n_parts);
        }

        std::vector<idx_t> xadj(n_cells + 1);
        for (size_t i = 0; i < n_cells + 1; ++i) {
            const auto off = static_cast<std::uint64_t>(cell2cell_offsets[i]);
            xadj[i] = checked_idx_cast(off, "cell adjacency offsets");
        }

        std::vector<idx_t> adjncy(cell2cell.size());
        for (size_t i = 0; i < cell2cell.size(); ++i) {
            const auto nbr = static_cast<std::uint64_t>(cell2cell[i]);
            adjncy[i] = checked_idx_cast(nbr, "cell adjacency");
        }

        idx_t nvtxs = checked_idx_cast(n_cells, "n_cells");
        idx_t ncon = 1;
        idx_t nparts = checked_idx_cast(static_cast<std::uint64_t>(n_parts), "n_parts");

        std::vector<idx_t> vwgt;
        idx_t* vwgt_ptr = nullptr;
        if (hint == svmp::PartitionHint::Vertices || hint == svmp::PartitionHint::Memory) {
            vwgt.resize(n_cells, 1);
            for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(n_cells); ++c) {
                auto [vptr, n] = mesh.cell_vertices_span(c);
                (void)vptr;

                std::uint64_t w = 1;
                if (hint == svmp::PartitionHint::Vertices) {
                    w = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(n));
                } else {
                    // Memory proxy similar to DistributedMesh::rebalance(Memory).
                    std::uint64_t mem = 0;
                    mem += sizeof(svmp::CellShape);
                    mem += static_cast<std::uint64_t>(n) * sizeof(svmp::index_t);
                    mem += sizeof(svmp::gid_t);
                    mem += sizeof(svmp::label_t);
                    mem += static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(mesh.dim()) *
                           sizeof(svmp::real_t) / 4u;
                    mem = static_cast<std::uint64_t>(static_cast<long double>(mem) * 1.2L);
                    w = std::max<std::uint64_t>(1, mem);
                }

                const auto max_idx = static_cast<std::uint64_t>(std::numeric_limits<idx_t>::max());
                if (w > max_idx) w = max_idx;
                vwgt[static_cast<size_t>(c)] = static_cast<idx_t>(w);
            }
            vwgt_ptr = vwgt.data();
        }

        std::vector<real_t> tpwgts(static_cast<size_t>(ncon) * static_cast<size_t>(nparts),
                                   static_cast<real_t>(1.0) / static_cast<real_t>(nparts));
        std::vector<real_t> ubvec(static_cast<size_t>(ncon), static_cast<real_t>(1.05));

        std::vector<idx_t> part(n_cells, 0);
        idx_t objval = 0;

        std::vector<idx_t> options(METIS_NOPTIONS, 0);
        METIS_SetDefaultOptions(options.data());
        options[METIS_OPTION_NUMBERING] = 0;
        options[METIS_OPTION_CONTIG] = 1;
        options[METIS_OPTION_SEED] = 42;

        const int status =
            METIS_PartGraphKway(&nvtxs,
                                &ncon,
                                xadj.data(),
                                adjncy.data(),
                                vwgt_ptr,
                                /*vsize=*/nullptr,
                                /*adjwgt=*/nullptr,
                                &nparts,
                                tpwgts.data(),
                                ubvec.data(),
                                options.data(),
                                &objval,
                                part.data());

        if (status != METIS_OK) {
            throw std::runtime_error("DistributedMesh: METIS_PartGraphKway failed");
        }

        std::vector<svmp::rank_t> cell_partition(n_cells, 0);
        for (size_t c = 0; c < n_cells; ++c) {
            const auto p = part[c];
            cell_partition[c] = static_cast<svmp::rank_t>(p);
        }
        return cell_partition;
    }
#endif
}
#endif

namespace svmp {

// ==========================================
// Helper functions for parallel I/O
// ==========================================

// Extract a submesh containing only cells assigned to a specific rank
static MeshBase extract_submesh(const MeshBase& global_mesh,
                               const std::vector<rank_t>& cell_partition,
                               rank_t target_rank,
                               const std::vector<index_t>* precomputed_cells = nullptr) {
    // =====================================================
    // Submesh extraction algorithm
    // =====================================================
    // Time complexity: O(n_cells + n_vertices)
    // Memory: O(cells_for_rank + vertices_for_rank)
    // This creates a new mesh containing only the cells
    // assigned to target_rank and their vertices

    MeshBase submesh;

    // Step 1: Identify cells for this rank
    std::vector<index_t> local_cells_storage;
    const std::vector<index_t>* local_cells_ptr = precomputed_cells;
    std::unordered_map<index_t, index_t> global_to_local_cell;

    if (!local_cells_ptr) {
        for (index_t c = 0; c < static_cast<index_t>(global_mesh.n_cells()); ++c) {
            if (cell_partition[static_cast<size_t>(c)] == target_rank) {
                local_cells_storage.push_back(c);
            }
        }
        local_cells_ptr = &local_cells_storage;
    }

    const auto& local_cells = *local_cells_ptr;
    global_to_local_cell.reserve(local_cells.size() * 2 + 1);
    for (index_t i = 0; i < static_cast<index_t>(local_cells.size()); ++i) {
        global_to_local_cell[local_cells[static_cast<size_t>(i)]] = i;
    }

    // Step 2: Identify vertices needed by these cells
    std::unordered_set<index_t> vertex_set;
    std::vector<index_t> local_vertices;
    std::unordered_map<index_t, index_t> global_to_local_vertex;

    for (index_t global_cell : local_cells) {
        auto [verts, n_verts] = global_mesh.cell_vertices_span(global_cell);
        for (size_t i = 0; i < n_verts; ++i) {
            if (vertex_set.insert(verts[i]).second) {
                global_to_local_vertex[verts[i]] = static_cast<index_t>(local_vertices.size());
                local_vertices.push_back(verts[i]);
            }
        }
    }

    // Step 3: Copy vertex coordinates
    std::vector<real_t> coords(local_vertices.size() * global_mesh.dim());
    const auto& global_coords = global_mesh.X_ref();
    for (size_t i = 0; i < local_vertices.size(); ++i) {
        index_t global_v = local_vertices[i];
        for (int d = 0; d < global_mesh.dim(); ++d) {
            coords[i * global_mesh.dim() + d] =
                global_coords[global_v * global_mesh.dim() + d];
        }
    }

    // Step 4: Build cell connectivity with local vertex indices
    std::vector<CellShape> shapes;
    std::vector<offset_t> offsets = {0};
    std::vector<index_t> connectivity;

    for (index_t global_cell : local_cells) {
        shapes.push_back(global_mesh.cell_shape(global_cell));

        auto [verts, n_verts] = global_mesh.cell_vertices_span(global_cell);
        for (size_t i = 0; i < n_verts; ++i) {
            connectivity.push_back(global_to_local_vertex[verts[i]]);
        }
        offsets.push_back(static_cast<offset_t>(connectivity.size()));
    }

    // Step 5: Copy global IDs
    std::vector<gid_t> vertex_gids(local_vertices.size());
    for (size_t i = 0; i < local_vertices.size(); ++i) {
        vertex_gids[i] = global_mesh.vertex_gids()[local_vertices[i]];
    }

    std::vector<gid_t> cell_gids(local_cells.size());
    for (size_t i = 0; i < local_cells.size(); ++i) {
        cell_gids[i] = global_mesh.cell_gids()[local_cells[i]];
    }

    // Step 6: Create submesh
    submesh.build_from_arrays(global_mesh.dim(), coords, offsets, connectivity, shapes);
    submesh.set_vertex_gids(std::move(vertex_gids));
    submesh.set_cell_gids(std::move(cell_gids));
    submesh.finalize();

    // Step 7: Copy field data if present
    // Preserve label registry (name <-> id) so downstream code can resolve labels.
    for (const auto& [label, name] : global_mesh.list_label_names()) {
        submesh.register_label(name, label);
    }

    // Copy per-cell region labels (critical for BC/material assignment).
    if (!global_mesh.cell_region_ids().empty()) {
        for (size_t i = 0; i < local_cells.size(); ++i) {
            const index_t global_c = local_cells[i];
            submesh.set_region_label(static_cast<index_t>(i), global_mesh.region_label(global_c));
        }
    }

    // Copy per-cell refinement levels when available.
    if (global_mesh.cell_refinement_levels().size() == global_mesh.n_cells()) {
        std::vector<size_t> levels;
        levels.reserve(local_cells.size());
        for (size_t i = 0; i < local_cells.size(); ++i) {
            levels.push_back(global_mesh.refinement_level(local_cells[i]));
        }
        submesh.set_cell_refinement_levels(std::move(levels));
    }

    // Copy vertex labels when available.
    if (global_mesh.vertex_label_ids().size() == global_mesh.n_vertices()) {
        for (size_t i = 0; i < local_vertices.size(); ++i) {
            const index_t global_v = local_vertices[i];
            submesh.set_vertex_label(static_cast<index_t>(i), global_mesh.vertex_label(global_v));
        }
    }

    // Copy entity sets for vertices/cells by mapping global indices -> local indices.
    for (const auto& set_name : global_mesh.list_sets(EntityKind::Vertex)) {
        const auto& ids = global_mesh.get_set(EntityKind::Vertex, set_name);
        for (const auto gv : ids) {
            const auto it = global_to_local_vertex.find(gv);
            if (it != global_to_local_vertex.end()) {
                submesh.add_to_set(EntityKind::Vertex, set_name, it->second);
            }
        }
    }
    for (const auto& set_name : global_mesh.list_sets(EntityKind::Volume)) {
        const auto& ids = global_mesh.get_set(EntityKind::Volume, set_name);
        for (const auto gc : ids) {
            const auto it = global_to_local_cell.find(gc);
            if (it != global_to_local_cell.end()) {
                submesh.add_to_set(EntityKind::Volume, set_name, it->second);
            }
        }
    }

    // Copy attached fields on vertices and cells.
    auto copy_field_kind = [&](EntityKind kind,
                               const std::vector<index_t>& global_ids,
                               const std::vector<gid_t>& global_entity_gids) {
        (void)global_entity_gids;
        const auto field_names = global_mesh.field_names(kind);
        if (field_names.empty()) {
            return;
        }

        for (const auto& name : field_names) {
            const auto src_handle = global_mesh.field_handle(kind, name);
            if (src_handle.id == 0) {
                continue;
            }

            const auto type = global_mesh.field_type(src_handle);
            const auto components = global_mesh.field_components(src_handle);
            const auto bytes_per_component = global_mesh.field_bytes_per_component_by_name(kind, name);
            const size_t bpe = components * bytes_per_component;

            const auto dst_handle = submesh.attach_field(kind, name, type, components, bytes_per_component);

            const auto* src = static_cast<const std::uint8_t*>(global_mesh.field_data(src_handle));
            auto* dst = static_cast<std::uint8_t*>(submesh.field_data(dst_handle));
            if (!src || !dst) {
                continue;
            }

            for (size_t i = 0; i < global_ids.size(); ++i) {
                const size_t global_idx = static_cast<size_t>(global_ids[i]);
                const size_t local_idx = i;
                std::memcpy(dst + local_idx * bpe, src + global_idx * bpe, bpe);
            }

            if (const auto* desc = global_mesh.field_descriptor(src_handle)) {
                submesh.set_field_descriptor(dst_handle, *desc);
            }
        }
    };

    copy_field_kind(EntityKind::Vertex, local_vertices, global_mesh.vertex_gids());
    copy_field_kind(EntityKind::Volume, local_cells, global_mesh.cell_gids());

    // ---- Copy codimension metadata (faces/edges) by canonical GID
    // These are required for production boundary-condition fidelity.
    ensure_canonical_face_gids(submesh);
    ensure_canonical_edge_gids(submesh);

    auto compute_canonical_face_gids = [&](const MeshBase& mesh) -> std::vector<gid_t> {
        std::vector<gid_t> gids(mesh.n_faces(), INVALID_GID);
        if (mesh.n_faces() == 0) {
            return gids;
        }
        const auto& vg = mesh.vertex_gids();
        if (vg.size() != mesh.n_vertices()) {
            return gids;
        }

        for (index_t f = 0; f < static_cast<index_t>(mesh.n_faces()); ++f) {
            auto [verts, n] = mesh.face_vertices_span(f);
            if (!verts || n == 0) continue;

            std::vector<gid_t> face_vertices;
            face_vertices.reserve(n);
            bool valid = true;
            for (size_t i = 0; i < n; ++i) {
                const auto v = verts[i];
                if (v < 0 || static_cast<size_t>(v) >= vg.size()) {
                    valid = false;
                    break;
                }
                const gid_t gid = vg[static_cast<size_t>(v)];
                if (gid == INVALID_GID) {
                    valid = false;
                    break;
                }
                face_vertices.push_back(gid);
            }
            if (!valid) continue;

            std::sort(face_vertices.begin(), face_vertices.end());
            face_vertices.erase(std::unique(face_vertices.begin(), face_vertices.end()), face_vertices.end());
            gids[static_cast<size_t>(f)] = stable_gid_from_sorted_vertex_gids(face_vertices);
        }
        return gids;
    };

    auto compute_canonical_edge_gids = [&](const MeshBase& mesh) -> std::vector<gid_t> {
        std::vector<gid_t> gids(mesh.n_edges(), INVALID_GID);
        if (mesh.n_edges() == 0) {
            return gids;
        }
        const auto& vg = mesh.vertex_gids();
        if (vg.size() != mesh.n_vertices()) {
            return gids;
        }
        const auto& e2v = mesh.edge2vertex();
        if (e2v.size() != mesh.n_edges()) {
            return gids;
        }

        for (index_t e = 0; e < static_cast<index_t>(mesh.n_edges()); ++e) {
            const auto v0 = e2v[static_cast<size_t>(e)][0];
            const auto v1 = e2v[static_cast<size_t>(e)][1];
            if (v0 < 0 || v1 < 0) continue;
            if (static_cast<size_t>(v0) >= vg.size() || static_cast<size_t>(v1) >= vg.size()) continue;

            const gid_t g0 = vg[static_cast<size_t>(v0)];
            const gid_t g1 = vg[static_cast<size_t>(v1)];
            if (g0 == INVALID_GID || g1 == INVALID_GID) continue;

            std::vector<gid_t> verts = {g0, g1};
            std::sort(verts.begin(), verts.end());
            verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
            gids[static_cast<size_t>(e)] = stable_gid_from_sorted_vertex_gids(verts);
        }
        return gids;
    };

    const auto global_face_gids = compute_canonical_face_gids(global_mesh);
    const auto global_edge_gids = compute_canonical_edge_gids(global_mesh);

    // Boundary face labels by canonical face GID.
    if (!global_mesh.face_boundary_ids().empty() &&
        global_mesh.face_boundary_ids().size() == global_mesh.n_faces() &&
        global_face_gids.size() == global_mesh.n_faces()) {
        std::unordered_map<gid_t, index_t> gid_to_global_face;
        gid_to_global_face.reserve(global_face_gids.size() * 2);
        for (index_t f = 0; f < static_cast<index_t>(global_face_gids.size()); ++f) {
            const gid_t gid = global_face_gids[static_cast<size_t>(f)];
            if (gid == INVALID_GID) continue;
            gid_to_global_face.emplace(gid, f);
        }

        for (index_t f = 0; f < static_cast<index_t>(submesh.n_faces()); ++f) {
            const gid_t gid = submesh.face_gids()[static_cast<size_t>(f)];
            const auto it = gid_to_global_face.find(gid);
            if (it == gid_to_global_face.end()) continue;
            const label_t label = global_mesh.face_boundary_ids()[static_cast<size_t>(it->second)];
            if (label == INVALID_LABEL) continue;
            submesh.set_boundary_label(f, label);
        }
    }

    // Edge labels by canonical edge GID.
    if (!global_mesh.edge_label_ids().empty() &&
        global_mesh.edge_label_ids().size() == global_mesh.n_edges() &&
        global_edge_gids.size() == global_mesh.n_edges()) {
        std::unordered_map<gid_t, index_t> gid_to_global_edge;
        gid_to_global_edge.reserve(global_edge_gids.size() * 2);
        for (index_t e = 0; e < static_cast<index_t>(global_edge_gids.size()); ++e) {
            const gid_t gid = global_edge_gids[static_cast<size_t>(e)];
            if (gid == INVALID_GID) continue;
            gid_to_global_edge.emplace(gid, e);
        }

        for (index_t e = 0; e < static_cast<index_t>(submesh.n_edges()); ++e) {
            const gid_t gid = submesh.edge_gids()[static_cast<size_t>(e)];
            const auto it = gid_to_global_edge.find(gid);
            if (it == gid_to_global_edge.end()) continue;
            const label_t label = global_mesh.edge_label_ids()[static_cast<size_t>(it->second)];
            if (label == INVALID_LABEL) continue;
            submesh.set_edge_label(e, label);
        }
    }

    // Sets (Face + Edge) by canonical GID.
    if (global_face_gids.size() == global_mesh.n_faces()) {
        for (const auto& set_name : global_mesh.list_sets(EntityKind::Face)) {
            const auto& ids = global_mesh.get_set(EntityKind::Face, set_name);
            for (const auto fid : ids) {
                if (fid < 0 || static_cast<size_t>(fid) >= global_face_gids.size()) continue;
                const gid_t gid = global_face_gids[static_cast<size_t>(fid)];
                if (gid == INVALID_GID) continue;
                const index_t local_f = submesh.global_to_local_face(gid);
                if (local_f == INVALID_INDEX) continue;
                submesh.add_to_set(EntityKind::Face, set_name, local_f);
            }
        }
    }

    if (global_edge_gids.size() == global_mesh.n_edges()) {
        for (const auto& set_name : global_mesh.list_sets(EntityKind::Edge)) {
            const auto& ids = global_mesh.get_set(EntityKind::Edge, set_name);
            for (const auto eid : ids) {
                if (eid < 0 || static_cast<size_t>(eid) >= global_edge_gids.size()) continue;
                const gid_t gid = global_edge_gids[static_cast<size_t>(eid)];
                if (gid == INVALID_GID) continue;
                const index_t local_e = submesh.global_to_local_edge(gid);
                if (local_e == INVALID_INDEX) continue;
                submesh.add_to_set(EntityKind::Edge, set_name, local_e);
            }
        }
    }

    // Fields (Face + Edge) by canonical GID.
    auto copy_gid_fields = [&](EntityKind kind,
                               const std::vector<gid_t>& global_gids,
                               const std::vector<gid_t>& local_gids) {
        if (global_gids.empty() || local_gids.empty()) {
            return;
        }

        std::unordered_map<gid_t, index_t> gid_to_global;
        gid_to_global.reserve(global_gids.size() * 2);
        for (index_t i = 0; i < static_cast<index_t>(global_gids.size()); ++i) {
            const gid_t gid = global_gids[static_cast<size_t>(i)];
            if (gid == INVALID_GID) continue;
            gid_to_global.emplace(gid, i);
        }

        for (const auto& name : global_mesh.field_names(kind)) {
            const auto src_handle = global_mesh.field_handle(kind, name);
            if (src_handle.id == 0) continue;

            const auto type = global_mesh.field_type(src_handle);
            const auto components = global_mesh.field_components(src_handle);
            size_t bytes_per_component = global_mesh.field_bytes_per_component_by_name(kind, name);
            if (bytes_per_component == 0) {
                bytes_per_component = bytes_per(type);
            }
            if (bytes_per_component == 0) continue;

            const size_t bpe = components * bytes_per_component;
            auto dst_handle = submesh.attach_field(kind, name, type, components, bytes_per_component);

            const auto* src = static_cast<const std::uint8_t*>(global_mesh.field_data(src_handle));
            auto* dst = static_cast<std::uint8_t*>(submesh.field_data(dst_handle));
            if (!src || !dst) continue;

            std::memset(dst, 0, submesh.field_entity_count(dst_handle) * bpe);
            for (index_t local_id = 0; local_id < static_cast<index_t>(local_gids.size()); ++local_id) {
                const gid_t gid = local_gids[static_cast<size_t>(local_id)];
                const auto it = gid_to_global.find(gid);
                if (it == gid_to_global.end()) continue;
                const index_t global_id = it->second;
                std::memcpy(dst + static_cast<size_t>(local_id) * bpe,
                            src + static_cast<size_t>(global_id) * bpe,
                            bpe);
            }

            if (const auto* desc = global_mesh.field_descriptor(src_handle)) {
                submesh.set_field_descriptor(dst_handle, *desc);
            }
        }
    };

    copy_gid_fields(EntityKind::Face, global_face_gids, submesh.face_gids());
    copy_gid_fields(EntityKind::Edge, global_edge_gids, submesh.edge_gids());

    return submesh;
}

// Serialize a mesh into a byte buffer for MPI communication
static void serialize_mesh(const MeshBase& mesh, std::vector<char>& buffer) {
    // =====================================================
    // Mesh serialization for MPI communication
    // =====================================================
    // Format: [header][vertices][cells][fields]
    // Time complexity: O(n_vertices + n_cells)
    // Memory: O(total_mesh_size)

    buffer.clear();

    // Reserve approximate size to avoid reallocations
    size_t estimated_size = sizeof(int) * 10 +  // headers
                           mesh.n_vertices() * mesh.dim() * sizeof(real_t) +
                           mesh.n_cells() * 10 * sizeof(index_t);  // estimate
    buffer.reserve(estimated_size);

    // Helper function to append data
    auto append_data = [&buffer](const void* data, size_t bytes) {
        if (bytes == 0) {
            return;
        }
        const char* ptr = static_cast<const char*>(data);
        buffer.insert(buffer.end(), ptr, ptr + bytes);
    };

    auto append_string = [&](const std::string& s) {
        const std::uint64_t len = static_cast<std::uint64_t>(s.size());
        append_data(&len, sizeof(std::uint64_t));
        if (len > 0) {
            append_data(s.data(), static_cast<size_t>(len));
        }
    };

    auto append_field_descriptor = [&](const FieldDescriptor& desc) {
        const int location = static_cast<int>(desc.location);
        const std::uint64_t components = static_cast<std::uint64_t>(desc.components);
        const real_t unit_scale = desc.unit_scale;
        const int time_dep = desc.time_dependent ? 1 : 0;
        const int intent = static_cast<int>(desc.intent);
        const int ghost_policy = static_cast<int>(desc.ghost_policy);

        append_data(&location, sizeof(int));
        append_data(&components, sizeof(std::uint64_t));
        append_data(&unit_scale, sizeof(real_t));
        append_data(&time_dep, sizeof(int));
        append_data(&intent, sizeof(int));
        append_data(&ghost_policy, sizeof(int));

        append_string(desc.units);
        append_string(desc.description);

        const std::uint64_t n_comp_names = static_cast<std::uint64_t>(desc.component_names.size());
        append_data(&n_comp_names, sizeof(std::uint64_t));
        for (const auto& name : desc.component_names) {
            append_string(name);
        }
    };

    // Header: dimensions and counts
    int dim = mesh.dim();
    int n_vertices = static_cast<int>(mesh.n_vertices());
    int n_cells = static_cast<int>(mesh.n_cells());

    append_data(&dim, sizeof(int));
    append_data(&n_vertices, sizeof(int));
    append_data(&n_cells, sizeof(int));

    // Vertex coordinates
    const auto& coords = mesh.X_ref();
    append_data(coords.data(), n_vertices * dim * sizeof(real_t));

    // Cell connectivity and shapes
    for (index_t c = 0; c < mesh.n_cells(); ++c) {
        // Serialize CellShape struct
        const auto& shape = mesh.cell_shape(c);
        int family = static_cast<int>(shape.family);
        append_data(&family, sizeof(int));
        append_data(&shape.num_corners, sizeof(int));
        append_data(&shape.order, sizeof(int));

        auto [verts, n_verts] = mesh.cell_vertices_span(c);
        int n = static_cast<int>(n_verts);
        append_data(&n, sizeof(int));
        append_data(verts, n_verts * sizeof(index_t));
    }

    // Global IDs
    append_data(mesh.vertex_gids().data(), n_vertices * sizeof(gid_t));
    append_data(mesh.cell_gids().data(), n_cells * sizeof(gid_t));

    // =====================================================
    // Metadata: labels, sets, and fields (required for production runs)
    // =====================================================

    // Cell region labels
    const bool has_cell_regions = (!mesh.cell_region_ids().empty() &&
                                  mesh.cell_region_ids().size() == mesh.n_cells());
    int has_regions_flag = has_cell_regions ? 1 : 0;
    append_data(&has_regions_flag, sizeof(int));
    if (has_cell_regions) {
        append_data(mesh.cell_region_ids().data(), mesh.n_cells() * sizeof(label_t));
    }

    // Vertex labels
    const bool has_vertex_labels = (!mesh.vertex_label_ids().empty() &&
                                   mesh.vertex_label_ids().size() == mesh.n_vertices());
    int has_vertex_labels_flag = has_vertex_labels ? 1 : 0;
    append_data(&has_vertex_labels_flag, sizeof(int));
    if (has_vertex_labels) {
        append_data(mesh.vertex_label_ids().data(), mesh.n_vertices() * sizeof(label_t));
    }

    // Cell refinement levels
    const bool has_refinement_levels =
        (!mesh.cell_refinement_levels().empty() &&
         mesh.cell_refinement_levels().size() == mesh.n_cells());
    int has_ref_levels_flag = has_refinement_levels ? 1 : 0;
    append_data(&has_ref_levels_flag, sizeof(int));
    if (has_refinement_levels) {
        std::vector<std::uint64_t> levels;
        levels.reserve(mesh.n_cells());
        for (const auto level : mesh.cell_refinement_levels()) {
            levels.push_back(static_cast<std::uint64_t>(level));
        }
        append_data(levels.data(), levels.size() * sizeof(std::uint64_t));
    }

    // Label registry (name <-> id)
    {
        const auto labels = mesh.list_label_names();
        const int n_labels = static_cast<int>(labels.size());
        append_data(&n_labels, sizeof(int));
        for (const auto& [label, name] : labels) {
            append_data(&label, sizeof(label_t));
            append_string(name);
        }
    }

    // Entity sets (Vertex + Volume)
    auto append_sets = [&](EntityKind kind) {
        const auto names = mesh.list_sets(kind);
        const int n_sets = static_cast<int>(names.size());
        append_data(&n_sets, sizeof(int));
        for (const auto& set_name : names) {
            append_string(set_name);

            const auto& ids = mesh.get_set(kind, set_name);
            const std::uint64_t n_ids = static_cast<std::uint64_t>(ids.size());
            append_data(&n_ids, sizeof(std::uint64_t));
            if (n_ids > 0) {
                append_data(ids.data(), static_cast<size_t>(n_ids) * sizeof(index_t));
            }
        }
    };
    append_sets(EntityKind::Vertex);
    append_sets(EntityKind::Volume);

    // Attached fields (Vertex + Volume only; face/edge fields require topology-aware transfer)
    struct FieldInfo {
        EntityKind kind;
        std::string name;
    };
    std::vector<FieldInfo> fields;
    for (const auto& name : mesh.field_names(EntityKind::Vertex)) {
        fields.push_back({EntityKind::Vertex, name});
    }
    for (const auto& name : mesh.field_names(EntityKind::Volume)) {
        fields.push_back({EntityKind::Volume, name});
    }

    const int n_fields = static_cast<int>(fields.size());
    append_data(&n_fields, sizeof(int));

    auto default_bytes_per_component = [](FieldScalarType t) -> size_t {
        switch (t) {
            case FieldScalarType::Float64: return sizeof(double);
            case FieldScalarType::Float32: return sizeof(float);
            case FieldScalarType::Int32: return sizeof(std::int32_t);
            case FieldScalarType::Int64: return sizeof(std::int64_t);
            case FieldScalarType::UInt8: return sizeof(std::uint8_t);
            case FieldScalarType::Custom: return 0;
        }
        return 0;
    };

    for (const auto& fi : fields) {
        const auto h = mesh.field_handle(fi.kind, fi.name);
        if (h.id == 0) {
            throw std::runtime_error("serialize_mesh: field '" + fi.name + "' missing handle");
        }

        const int kind_i = static_cast<int>(fi.kind);
        const int type_i = static_cast<int>(mesh.field_type(h));
        const std::uint64_t components = static_cast<std::uint64_t>(mesh.field_components(h));

        size_t bytes_per_component = mesh.field_bytes_per_component_by_name(fi.kind, fi.name);
        if (bytes_per_component == 0) {
            bytes_per_component = default_bytes_per_component(mesh.field_type(h));
        }
        if (bytes_per_component == 0) {
            throw std::runtime_error("serialize_mesh: could not determine bytes_per_component for field '" +
                                     fi.name + "'");
        }

        const std::uint64_t bpc = static_cast<std::uint64_t>(bytes_per_component);
        append_data(&kind_i, sizeof(int));
        append_data(&type_i, sizeof(int));
        append_data(&components, sizeof(std::uint64_t));
        append_data(&bpc, sizeof(std::uint64_t));
        append_string(fi.name);

        const std::uint64_t n_entities = static_cast<std::uint64_t>(mesh.field_entity_count(h));
        const std::uint64_t bytes_per_entity =
            static_cast<std::uint64_t>(mesh.field_bytes_per_entity(h));
        const std::uint64_t total_bytes = n_entities * bytes_per_entity;
        append_data(&n_entities, sizeof(std::uint64_t));
        append_data(&total_bytes, sizeof(std::uint64_t));

        const auto* src = static_cast<const std::uint8_t*>(mesh.field_data(h));
        if (total_bytes > 0 && !src) {
            throw std::runtime_error("serialize_mesh: null field data for '" + fi.name + "'");
        }
        if (total_bytes > 0) {
            append_data(src, static_cast<size_t>(total_bytes));
        }

        int has_descriptor = 0;
        if (const auto* desc = mesh.field_descriptor(h)) {
            has_descriptor = 1;
            append_data(&has_descriptor, sizeof(int));
            append_field_descriptor(*desc);
        } else {
            append_data(&has_descriptor, sizeof(int));
        }
    }

    // =====================================================
    // Codimension entities: faces/edges (boundary labels, sets, fields)
    // =====================================================
    // These entities are reconstructed on the receiver by MeshBase::finalize().
    // Do not serialize by local index alone; key metadata by stable canonical GIDs.

    const auto compute_canonical_face_gids = [&]() -> std::vector<gid_t> {
        std::vector<gid_t> gids(mesh.n_faces(), INVALID_GID);
        if (mesh.n_faces() == 0) {
            return gids;
        }
        const auto& vg = mesh.vertex_gids();
        if (vg.size() != mesh.n_vertices()) {
            return gids;
        }

        for (index_t f = 0; f < static_cast<index_t>(mesh.n_faces()); ++f) {
            auto [verts, n] = mesh.face_vertices_span(f);
            if (!verts || n == 0) continue;
            std::vector<gid_t> face_vertices;
            face_vertices.reserve(n);
            bool valid = true;
            for (size_t i = 0; i < n; ++i) {
                const auto v = verts[i];
                if (v < 0 || static_cast<size_t>(v) >= vg.size()) {
                    valid = false;
                    break;
                }
                const auto gid = vg[static_cast<size_t>(v)];
                if (gid == INVALID_GID) {
                    valid = false;
                    break;
                }
                face_vertices.push_back(gid);
            }
            if (!valid) continue;
            std::sort(face_vertices.begin(), face_vertices.end());
            face_vertices.erase(std::unique(face_vertices.begin(), face_vertices.end()),
                                face_vertices.end());
            gids[static_cast<size_t>(f)] = stable_gid_from_sorted_vertex_gids(face_vertices);
        }
        return gids;
    };

    const auto compute_canonical_edge_gids = [&]() -> std::vector<gid_t> {
        std::vector<gid_t> gids(mesh.n_edges(), INVALID_GID);
        if (mesh.n_edges() == 0) {
            return gids;
        }
        const auto& vg = mesh.vertex_gids();
        if (vg.size() != mesh.n_vertices()) {
            return gids;
        }

        for (index_t e = 0; e < static_cast<index_t>(mesh.n_edges()); ++e) {
            const auto ev = mesh.edge_vertices(e);
            const auto v0 = ev[0];
            const auto v1 = ev[1];
            if (v0 < 0 || v1 < 0) continue;
            if (static_cast<size_t>(v0) >= vg.size() || static_cast<size_t>(v1) >= vg.size()) continue;
            const auto g0 = vg[static_cast<size_t>(v0)];
            const auto g1 = vg[static_cast<size_t>(v1)];
            if (g0 == INVALID_GID || g1 == INVALID_GID) continue;

            std::vector<gid_t> verts = {g0, g1};
            std::sort(verts.begin(), verts.end());
            verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
            gids[static_cast<size_t>(e)] = stable_gid_from_sorted_vertex_gids(verts);
        }
        return gids;
    };

    const auto face_gids = compute_canonical_face_gids();
    const auto edge_gids = compute_canonical_edge_gids();

    const std::uint64_t n_faces = static_cast<std::uint64_t>(face_gids.size());
    append_data(&n_faces, sizeof(std::uint64_t));
    if (n_faces > 0) {
        append_data(face_gids.data(), static_cast<size_t>(n_faces) * sizeof(gid_t));
    }

    const std::uint64_t n_edges = static_cast<std::uint64_t>(edge_gids.size());
    append_data(&n_edges, sizeof(std::uint64_t));
    if (n_edges > 0) {
        append_data(edge_gids.data(), static_cast<size_t>(n_edges) * sizeof(gid_t));
    }

    // Boundary labels per face (sparse: only non-INVALID_LABEL entries).
    std::vector<std::pair<gid_t, label_t>> boundary_labels;
    const auto& face_boundary = mesh.face_boundary_ids();
    if (!face_boundary.empty() && face_boundary.size() == mesh.n_faces() && face_gids.size() == mesh.n_faces()) {
        for (size_t f = 0; f < face_boundary.size(); ++f) {
            const auto label = face_boundary[f];
            const auto gid = face_gids[f];
            if (label == INVALID_LABEL || gid == INVALID_GID) continue;
            boundary_labels.push_back({gid, label});
        }
    }
    const std::uint64_t n_boundary_labels = static_cast<std::uint64_t>(boundary_labels.size());
    append_data(&n_boundary_labels, sizeof(std::uint64_t));
    for (const auto& [gid, label] : boundary_labels) {
        append_data(&gid, sizeof(gid_t));
        append_data(&label, sizeof(label_t));
    }

    // Edge labels (sparse: only non-INVALID_LABEL entries).
    std::vector<std::pair<gid_t, label_t>> edge_labels;
    const auto& edge_label_ids = mesh.edge_label_ids();
    if (!edge_label_ids.empty() && edge_label_ids.size() == mesh.n_edges() && edge_gids.size() == mesh.n_edges()) {
        for (size_t e = 0; e < edge_label_ids.size(); ++e) {
            const auto label = edge_label_ids[e];
            const auto gid = edge_gids[e];
            if (label == INVALID_LABEL || gid == INVALID_GID) continue;
            edge_labels.push_back({gid, label});
        }
    }
    const std::uint64_t n_edge_labels = static_cast<std::uint64_t>(edge_labels.size());
    append_data(&n_edge_labels, sizeof(std::uint64_t));
    for (const auto& [gid, label] : edge_labels) {
        append_data(&gid, sizeof(gid_t));
        append_data(&label, sizeof(label_t));
    }

    // Sets (Face + Edge) serialized as GID lists (robust to face/edge index permutations).
    auto append_gid_sets = [&](EntityKind kind, const std::vector<gid_t>& gids) {
        const auto names = mesh.list_sets(kind);
        const int n_sets = static_cast<int>(names.size());
        append_data(&n_sets, sizeof(int));
        for (const auto& set_name : names) {
            append_string(set_name);

            const auto& ids = mesh.get_set(kind, set_name);
            std::vector<gid_t> members;
            members.reserve(ids.size());
            for (const auto id : ids) {
                if (id < 0 || static_cast<size_t>(id) >= gids.size()) continue;
                const auto gid = gids[static_cast<size_t>(id)];
                if (gid == INVALID_GID) continue;
                members.push_back(gid);
            }

            const std::uint64_t n_ids = static_cast<std::uint64_t>(members.size());
            append_data(&n_ids, sizeof(std::uint64_t));
            if (n_ids > 0) {
                append_data(members.data(), static_cast<size_t>(n_ids) * sizeof(gid_t));
            }
        }
    };
    append_gid_sets(EntityKind::Face, face_gids);
    append_gid_sets(EntityKind::Edge, edge_gids);

    // Fields (Face + Edge) serialized as (gid[i], bytes[i]) tuples using the canonical GID arrays.
    auto append_gid_fields = [&](EntityKind kind, const std::vector<gid_t>& gids) {
        const auto names = mesh.field_names(kind);
        const int n_kind_fields = static_cast<int>(names.size());
        append_data(&n_kind_fields, sizeof(int));
        for (const auto& field_name : names) {
            const auto h = mesh.field_handle(kind, field_name);
            if (h.id == 0) {
                throw std::runtime_error("serialize_mesh: missing handle for field '" + field_name + "'");
            }

            const int type_i = static_cast<int>(mesh.field_type(h));
            const std::uint64_t components = static_cast<std::uint64_t>(mesh.field_components(h));

            size_t bytes_per_component = mesh.field_bytes_per_component_by_name(kind, field_name);
            if (bytes_per_component == 0) {
                bytes_per_component = default_bytes_per_component(mesh.field_type(h));
            }
            if (bytes_per_component == 0) {
                throw std::runtime_error("serialize_mesh: could not determine bytes_per_component for field '" +
                                         field_name + "'");
            }

            const std::uint64_t bpc = static_cast<std::uint64_t>(bytes_per_component);
            append_data(&type_i, sizeof(int));
            append_data(&components, sizeof(std::uint64_t));
            append_data(&bpc, sizeof(std::uint64_t));
            append_string(field_name);

            int has_descriptor = 0;
            if (const auto* desc = mesh.field_descriptor(h)) {
                has_descriptor = 1;
                append_data(&has_descriptor, sizeof(int));
                append_field_descriptor(*desc);
            } else {
                append_data(&has_descriptor, sizeof(int));
            }

            const std::uint64_t n_entities = static_cast<std::uint64_t>(mesh.field_entity_count(h));
            const std::uint64_t bytes_per_entity = static_cast<std::uint64_t>(mesh.field_bytes_per_entity(h));
            const std::uint64_t total_bytes = n_entities * bytes_per_entity;
            append_data(&n_entities, sizeof(std::uint64_t));
            append_data(&total_bytes, sizeof(std::uint64_t));

            if (n_entities != static_cast<std::uint64_t>(gids.size())) {
                throw std::runtime_error("serialize_mesh: field '" + field_name + "' entity count mismatch");
            }

            const auto* src = static_cast<const std::uint8_t*>(mesh.field_data(h));
            if (total_bytes > 0 && !src) {
                throw std::runtime_error("serialize_mesh: null field data for '" + field_name + "'");
            }
            if (total_bytes > 0) {
                append_data(src, static_cast<size_t>(total_bytes));
            }
        }
    };

    append_gid_fields(EntityKind::Face, face_gids);
    append_gid_fields(EntityKind::Edge, edge_gids);
}

// Deserialize a mesh from a byte buffer received via MPI
static void deserialize_mesh(const std::vector<char>& buffer, MeshBase& mesh) {
    // =====================================================
    // Mesh deserialization from MPI buffer
    // =====================================================
    // Time complexity: O(n_vertices + n_cells)
    // Memory: O(mesh_size)

    size_t offset = 0;

    // Helper function to read data
    auto read_data = [&buffer, &offset](void* dest, size_t bytes) {
        if (offset + bytes > buffer.size()) {
            throw std::runtime_error("deserialize_mesh: truncated buffer");
        }
        std::memcpy(dest, buffer.data() + offset, bytes);
        offset += bytes;
    };

    auto read_string = [&]() -> std::string {
        std::uint64_t len = 0;
        read_data(&len, sizeof(std::uint64_t));
        std::string s;
        s.resize(static_cast<size_t>(len));
        if (len > 0) {
            read_data(s.data(), static_cast<size_t>(len));
        }
        return s;
    };

    auto read_field_descriptor = [&]() -> FieldDescriptor {
        FieldDescriptor desc;
        int location = 0;
        std::uint64_t components = 0;
        real_t unit_scale = 1.0;
        int time_dep = 0;
        int intent = 0;
        int ghost_policy = 0;

        read_data(&location, sizeof(int));
        read_data(&components, sizeof(std::uint64_t));
        read_data(&unit_scale, sizeof(real_t));
        read_data(&time_dep, sizeof(int));
        read_data(&intent, sizeof(int));
        read_data(&ghost_policy, sizeof(int));

        desc.location = static_cast<EntityKind>(location);
        desc.components = static_cast<size_t>(components);
        desc.unit_scale = unit_scale;
        desc.time_dependent = (time_dep != 0);
        desc.intent = static_cast<FieldIntent>(intent);
        desc.ghost_policy = static_cast<FieldGhostPolicy>(ghost_policy);

        desc.units = read_string();
        desc.description = read_string();

        std::uint64_t n_comp_names = 0;
        read_data(&n_comp_names, sizeof(std::uint64_t));
        desc.component_names.clear();
        desc.component_names.reserve(static_cast<size_t>(n_comp_names));
        for (std::uint64_t i = 0; i < n_comp_names; ++i) {
            desc.component_names.push_back(read_string());
        }

        return desc;
    };

    // Read header
    int dim, n_vertices, n_cells;
    read_data(&dim, sizeof(int));
    read_data(&n_vertices, sizeof(int));
    read_data(&n_cells, sizeof(int));

    // Read vertex coordinates
    std::vector<real_t> coords(n_vertices * dim);
    read_data(coords.data(), n_vertices * dim * sizeof(real_t));

    // Read cell connectivity and shapes
    std::vector<CellShape> shapes;
    std::vector<offset_t> offsets = {0};
    std::vector<index_t> connectivity;

    for (int c = 0; c < n_cells; ++c) {
        // Deserialize CellShape struct
        CellShape shape;
        int family;
        read_data(&family, sizeof(int));
        shape.family = static_cast<CellFamily>(family);
        read_data(&shape.num_corners, sizeof(int));
        read_data(&shape.order, sizeof(int));
        shapes.push_back(shape);

        int n_verts;
        read_data(&n_verts, sizeof(int));
        std::vector<index_t> cell_verts(n_verts);
        read_data(cell_verts.data(), n_verts * sizeof(index_t));
        connectivity.insert(connectivity.end(), cell_verts.begin(), cell_verts.end());
        offsets.push_back(static_cast<offset_t>(connectivity.size()));
    }

    // Read global IDs
    std::vector<gid_t> vertex_gids(n_vertices);
    std::vector<gid_t> cell_gids(n_cells);
    read_data(vertex_gids.data(), n_vertices * sizeof(gid_t));
    read_data(cell_gids.data(), n_cells * sizeof(gid_t));

    // Initialize mesh
    mesh.build_from_arrays(dim, coords, offsets, connectivity, shapes);
    mesh.set_vertex_gids(std::move(vertex_gids));
    mesh.set_cell_gids(std::move(cell_gids));

    // Backward compatibility: older buffers ended after GIDs.
    if (offset == buffer.size()) {
        mesh.finalize();
        return;
    }

    // =====================================================
    // Metadata: labels, sets, and fields
    // =====================================================

    // Cell region labels
    int has_regions_flag = 0;
    read_data(&has_regions_flag, sizeof(int));
    if (has_regions_flag) {
        std::vector<label_t> regions(static_cast<size_t>(n_cells));
        read_data(regions.data(), static_cast<size_t>(n_cells) * sizeof(label_t));
        for (int c = 0; c < n_cells; ++c) {
            mesh.set_region_label(static_cast<index_t>(c), regions[static_cast<size_t>(c)]);
        }
    }

    // Vertex labels
    int has_vertex_labels_flag = 0;
    read_data(&has_vertex_labels_flag, sizeof(int));
    if (has_vertex_labels_flag) {
        std::vector<label_t> vlabels(static_cast<size_t>(n_vertices));
        read_data(vlabels.data(), static_cast<size_t>(n_vertices) * sizeof(label_t));
        for (int v = 0; v < n_vertices; ++v) {
            mesh.set_vertex_label(static_cast<index_t>(v), vlabels[static_cast<size_t>(v)]);
        }
    }

    // Cell refinement levels
    int has_ref_levels_flag = 0;
    read_data(&has_ref_levels_flag, sizeof(int));
    if (has_ref_levels_flag) {
        std::vector<std::uint64_t> levels64(static_cast<size_t>(n_cells));
        read_data(levels64.data(), static_cast<size_t>(n_cells) * sizeof(std::uint64_t));
        std::vector<size_t> levels;
        levels.reserve(static_cast<size_t>(n_cells));
        for (const auto v : levels64) {
            levels.push_back(static_cast<size_t>(v));
        }
        mesh.set_cell_refinement_levels(std::move(levels));
    }

    // Label registry
    {
        int n_labels = 0;
        read_data(&n_labels, sizeof(int));
        for (int i = 0; i < n_labels; ++i) {
            label_t label = INVALID_LABEL;
            read_data(&label, sizeof(label_t));
            std::string name = read_string();
            mesh.register_label(name, label);
        }
    }

    // Entity sets (Vertex + Volume)
    auto read_sets = [&](EntityKind kind) {
        int n_sets = 0;
        read_data(&n_sets, sizeof(int));
        for (int s = 0; s < n_sets; ++s) {
            std::string set_name = read_string();

            std::uint64_t n_ids = 0;
            read_data(&n_ids, sizeof(std::uint64_t));
            std::vector<index_t> ids(static_cast<size_t>(n_ids));
            if (n_ids > 0) {
                read_data(ids.data(), static_cast<size_t>(n_ids) * sizeof(index_t));
            }
            for (const auto id : ids) {
                mesh.add_to_set(kind, set_name, id);
            }
        }
    };
    read_sets(EntityKind::Vertex);
    read_sets(EntityKind::Volume);

    // Fields (Vertex + Volume)
    int n_fields = 0;
    read_data(&n_fields, sizeof(int));
    for (int f = 0; f < n_fields; ++f) {
        int kind_i = 0;
        int type_i = 0;
        std::uint64_t components = 0;
        std::uint64_t bytes_per_component = 0;
        read_data(&kind_i, sizeof(int));
        read_data(&type_i, sizeof(int));
        read_data(&components, sizeof(std::uint64_t));
        read_data(&bytes_per_component, sizeof(std::uint64_t));
        std::string name = read_string();

        std::uint64_t n_entities = 0;
        std::uint64_t total_bytes = 0;
        read_data(&n_entities, sizeof(std::uint64_t));
        read_data(&total_bytes, sizeof(std::uint64_t));

        const auto kind = static_cast<EntityKind>(kind_i);
        const auto type = static_cast<FieldScalarType>(type_i);

        const size_t expected_entities =
            (kind == EntityKind::Vertex) ? static_cast<size_t>(n_vertices)
                                         : (kind == EntityKind::Volume) ? static_cast<size_t>(n_cells)
                                                                        : 0u;
        if (expected_entities != static_cast<size_t>(n_entities)) {
            throw std::runtime_error("deserialize_mesh: field '" + name + "' has unexpected entity count");
        }

        const auto h = mesh.attach_field(kind, name, type, static_cast<size_t>(components),
                                         static_cast<size_t>(bytes_per_component));
        auto* dst = static_cast<std::uint8_t*>(mesh.field_data(h));
        const size_t expected_bytes = mesh.field_entity_count(h) * mesh.field_bytes_per_entity(h);
        if (static_cast<size_t>(total_bytes) != expected_bytes) {
            throw std::runtime_error("deserialize_mesh: field '" + name + "' size mismatch");
        }
        if (expected_bytes > 0 && !dst) {
            throw std::runtime_error("deserialize_mesh: null destination field buffer");
        }
        if (expected_bytes > 0) {
            read_data(dst, expected_bytes);
        }

        int has_descriptor = 0;
        read_data(&has_descriptor, sizeof(int));
        if (has_descriptor) {
            const auto desc = read_field_descriptor();
            mesh.set_field_descriptor(h, desc);
        }
    }

    // Finalize before codim-1/codim-2 metadata application.
    mesh.finalize();

    // Ensure face/edge GIDs are stable so we can map by canonical ID.
    ensure_canonical_face_gids(mesh);
    ensure_canonical_edge_gids(mesh);

    // =====================================================
    // Codimension entities: faces/edges (boundary labels, sets, fields)
    // =====================================================

    std::uint64_t n_faces = 0;
    read_data(&n_faces, sizeof(std::uint64_t));
    std::vector<gid_t> face_gids(static_cast<size_t>(n_faces));
    if (n_faces > 0) {
        read_data(face_gids.data(), static_cast<size_t>(n_faces) * sizeof(gid_t));
    }
    if (n_faces != static_cast<std::uint64_t>(mesh.n_faces())) {
        throw std::runtime_error("deserialize_mesh: face count mismatch");
    }

    std::uint64_t n_edges = 0;
    read_data(&n_edges, sizeof(std::uint64_t));
    std::vector<gid_t> edge_gids(static_cast<size_t>(n_edges));
    if (n_edges > 0) {
        read_data(edge_gids.data(), static_cast<size_t>(n_edges) * sizeof(gid_t));
    }
    if (n_edges != static_cast<std::uint64_t>(mesh.n_edges())) {
        throw std::runtime_error("deserialize_mesh: edge count mismatch");
    }

    // Face boundary labels
    std::uint64_t n_boundary_labels = 0;
    read_data(&n_boundary_labels, sizeof(std::uint64_t));
    for (std::uint64_t i = 0; i < n_boundary_labels; ++i) {
        gid_t gid = INVALID_GID;
        label_t label = INVALID_LABEL;
        read_data(&gid, sizeof(gid_t));
        read_data(&label, sizeof(label_t));
        const index_t f = mesh.global_to_local_face(gid);
        if (f == INVALID_INDEX) continue;
        mesh.set_boundary_label(f, label);
    }

    // Edge labels
    std::uint64_t n_edge_labels = 0;
    read_data(&n_edge_labels, sizeof(std::uint64_t));
    for (std::uint64_t i = 0; i < n_edge_labels; ++i) {
        gid_t gid = INVALID_GID;
        label_t label = INVALID_LABEL;
        read_data(&gid, sizeof(gid_t));
        read_data(&label, sizeof(label_t));
        const index_t e = mesh.global_to_local_edge(gid);
        if (e == INVALID_INDEX) continue;
        mesh.set_edge_label(e, label);
    }

    // Sets (Face + Edge) stored as GID lists.
    auto read_gid_sets = [&](EntityKind kind, auto gid_to_local) {
        int n_sets = 0;
        read_data(&n_sets, sizeof(int));
        for (int s = 0; s < n_sets; ++s) {
            const std::string set_name = read_string();
            std::uint64_t n_ids = 0;
            read_data(&n_ids, sizeof(std::uint64_t));
            std::vector<gid_t> gids(static_cast<size_t>(n_ids));
            if (n_ids > 0) {
                read_data(gids.data(), static_cast<size_t>(n_ids) * sizeof(gid_t));
            }
            for (const auto gid : gids) {
                const index_t id = gid_to_local(gid);
                if (id == INVALID_INDEX) continue;
                mesh.add_to_set(kind, set_name, id);
            }
        }
    };
    read_gid_sets(EntityKind::Face, [&](gid_t gid) { return mesh.global_to_local_face(gid); });
    read_gid_sets(EntityKind::Edge, [&](gid_t gid) { return mesh.global_to_local_edge(gid); });

    // Fields (Face + Edge) stored as raw bytes in sender index order, remapped by canonical GIDs.
    auto read_gid_fields = [&](EntityKind kind,
                               const std::vector<gid_t>& gids,
                               auto gid_to_local) {
        int n_kind_fields = 0;
        read_data(&n_kind_fields, sizeof(int));
        for (int f = 0; f < n_kind_fields; ++f) {
            int type_i = 0;
            std::uint64_t components = 0;
            std::uint64_t bytes_per_component = 0;
            read_data(&type_i, sizeof(int));
            read_data(&components, sizeof(std::uint64_t));
            read_data(&bytes_per_component, sizeof(std::uint64_t));
            const std::string name = read_string();

            int has_descriptor = 0;
            read_data(&has_descriptor, sizeof(int));
            FieldDescriptor desc;
            if (has_descriptor) {
                desc = read_field_descriptor();
            }

            std::uint64_t n_entities = 0;
            std::uint64_t total_bytes = 0;
            read_data(&n_entities, sizeof(std::uint64_t));
            read_data(&total_bytes, sizeof(std::uint64_t));

            if (n_entities != static_cast<std::uint64_t>(gids.size())) {
                throw std::runtime_error("deserialize_mesh: field '" + name + "' entity count mismatch");
            }

            const auto type = static_cast<FieldScalarType>(type_i);
            const auto h = mesh.attach_field(kind, name, type, static_cast<size_t>(components),
                                             static_cast<size_t>(bytes_per_component));

            const size_t bytes_per_entity = mesh.field_bytes_per_entity(h);
            const size_t expected_bytes = mesh.field_entity_count(h) * bytes_per_entity;
            if (static_cast<size_t>(total_bytes) != expected_bytes) {
                throw std::runtime_error("deserialize_mesh: field '" + name + "' size mismatch");
            }

            std::vector<std::uint8_t> tmp(expected_bytes);
            if (expected_bytes > 0) {
                read_data(tmp.data(), expected_bytes);
            }

            auto* dst = static_cast<std::uint8_t*>(mesh.field_data(h));
            if (!dst && expected_bytes > 0) {
                throw std::runtime_error("deserialize_mesh: null destination for field '" + name + "'");
            }
            if (dst && expected_bytes > 0) {
                std::memset(dst, 0, expected_bytes);
            }

            for (size_t i = 0; i < gids.size(); ++i) {
                const auto gid = gids[i];
                const index_t local = gid_to_local(gid);
                if (local == INVALID_INDEX) continue;
                const size_t lid = static_cast<size_t>(local);
                if (lid >= mesh.field_entity_count(h)) continue;
                std::memcpy(dst + lid * bytes_per_entity,
                            tmp.data() + i * bytes_per_entity,
                            bytes_per_entity);
            }

            if (has_descriptor) {
                mesh.set_field_descriptor(h, desc);
            }
        }
    };

    read_gid_fields(EntityKind::Face, face_gids, [&](gid_t gid) { return mesh.global_to_local_face(gid); });
    read_gid_fields(EntityKind::Edge, edge_gids, [&](gid_t gid) { return mesh.global_to_local_edge(gid); });
}

#if defined(MESH_BUILD_TESTS)
namespace test::internal {
void serialize_mesh_for_test(const MeshBase& mesh, std::vector<char>& buffer) { serialize_mesh(mesh, buffer); }
void deserialize_mesh_for_test(const std::vector<char>& buffer, MeshBase& mesh) { deserialize_mesh(buffer, mesh); }
} // namespace test::internal
#endif

// Tree-based mesh distribution for large-scale runs
static void distribute_mesh_tree(const MeshBase& global_mesh,
                                const std::vector<rank_t>& cell_partition,
                                MPI_Comm comm) {
    // =====================================================
    // Binary tree distribution pattern
    // =====================================================
    // Algorithm: Each rank splits its data and sends half to a child
    // Time complexity: O(n * log(p)) total
    // Memory: O(n/2^level) at each tree level
    // Scaling: Reduces root bottleneck for large rank counts

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        // Root initiates tree distribution
        int tree_height = static_cast<int>(std::log2(size)) + 1;

        for (int level = 0; level < tree_height; ++level) {
            int step = 1 << (level + 1);  // 2^(level+1)
            int sender_mask = (1 << level) - 1;

            if ((rank & sender_mask) == 0 && rank + (step/2) < size) {
                // This rank sends to rank + step/2
                int target = rank + (step/2);

                // Extract submesh for all ranks >= target
                // that this sender is responsible for
                MeshBase submesh_to_send = extract_submesh(global_mesh, cell_partition, target);

                // Serialize and send
                std::vector<char> buffer;
                serialize_mesh(submesh_to_send, buffer);

                int buffer_size = static_cast<int>(buffer.size());
                MPI_Send(&buffer_size, 1, MPI_INT, target, level * 2, comm);
                MPI_Send(buffer.data(), buffer_size, MPI_CHAR, target, level * 2 + 1, comm);
            }
        }
    }
}

// Receive mesh in tree-based distribution
static void receive_mesh_tree(MeshBase& mesh, MPI_Comm comm) {
    // =====================================================
    // Binary tree reception pattern
    // =====================================================
    // Receives mesh data from parent in tree hierarchy

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank != 0) {
        // Determine which level this rank receives at
        int level = static_cast<int>(std::log2(rank & -rank));

        // Calculate sender
        int step = 1 << level;
        int sender = rank - step;

        // Receive buffer size
        int buffer_size;
        MPI_Recv(&buffer_size, 1, MPI_INT, sender, level * 2, comm, MPI_STATUS_IGNORE);

        // Receive buffer
        std::vector<char> buffer(buffer_size);
        MPI_Recv(buffer.data(), buffer_size, MPI_CHAR, sender, level * 2 + 1, comm, MPI_STATUS_IGNORE);

        // Deserialize
        deserialize_mesh(buffer, mesh);
    }
}

// ==========================================
// Constructors
// ==========================================

DistributedMesh::DistributedMesh()
    : local_mesh_(std::make_shared<MeshBase>()) {
    set_mpi_comm(MPI_COMM_SELF);
}

DistributedMesh::DistributedMesh(MPI_Comm comm)
    : local_mesh_(std::make_shared<MeshBase>()) {
    set_mpi_comm(comm);
}

DistributedMesh::DistributedMesh(std::shared_ptr<MeshBase> local_mesh, MPI_Comm comm)
    : local_mesh_(local_mesh ? local_mesh : std::make_shared<MeshBase>()) {

    if (!local_mesh_) {
        throw std::invalid_argument("DistributedMesh: null local mesh provided");
    }

    set_mpi_comm(comm);

    // Initialize ownership arrays
    vertex_owner_.resize(local_mesh_->n_vertices(), Ownership::Owned);
    cell_owner_.resize(local_mesh_->n_cells(), Ownership::Owned);
    face_owner_.resize(local_mesh_->n_faces(), Ownership::Owned);
    edge_owner_.resize(local_mesh_->n_edges(), Ownership::Owned);

    vertex_owner_rank_.resize(local_mesh_->n_vertices(), my_rank_);
    cell_owner_rank_.resize(local_mesh_->n_cells(), my_rank_);
    face_owner_rank_.resize(local_mesh_->n_faces(), my_rank_);
    edge_owner_rank_.resize(local_mesh_->n_edges(), my_rank_);
}

// ==========================================
// Builders / Construction
// ==========================================

	void DistributedMesh::build_from_arrays_global_and_partition(
		    int spatial_dim,
		    const std::vector<real_t>& X_ref,
		    const std::vector<offset_t>& cell2vertex_offsets,
		    const std::vector<index_t>& cell2vertex,
	    const std::vector<CellShape>& cell_shape,
	    PartitionHint hint,
	    int ghost_layers,
	    const std::unordered_map<std::string, std::string>& options) {
	#ifdef MESH_HAS_MPI
	    if (!local_mesh_) {
	        local_mesh_ = std::make_shared<MeshBase>();
	    }

    // Serial / non-initialized MPI path.
    if (world_size_ <= 1) {
        build_from_arrays_local(spatial_dim, X_ref, cell2vertex_offsets, cell2vertex, cell_shape);
        return;
    }

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        throw std::runtime_error(
            "DistributedMesh::build_from_arrays_global_and_partition: MPI is not initialized");
    }
    if (comm_ == MPI_COMM_NULL) {
        throw std::runtime_error(
            "DistributedMesh::build_from_arrays_global_and_partition: MPI communicator is null");
    }

    if (ghost_layers < 0) {
        throw std::invalid_argument(
            "DistributedMesh::build_from_arrays_global_and_partition: ghost_layers must be >= 0");
    }

    std::string partition_method;
    const auto it = options.find("partition_method");
    if (it != options.end()) {
        partition_method = it->second;
        std::transform(partition_method.begin(), partition_method.end(),
                       partition_method.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    } else {
        // Default: use graph partitioning whenever a multi-rank build requests
        // a non-trivial partition hint (Cells/Vertices/Memory/Metis/ParMetis).
        if (hint == PartitionHint::None) {
            partition_method = "block";
        } else if (hint == PartitionHint::ParMetis) {
            partition_method = "parmetis";
	        } else {
	            partition_method = "metis";
	        }
	    }

    // Scalable ParMETIS startup path:
    // 1) scatter connectivity (no coordinates) by block
    // 2) ParMETIS partitions cells based on global vertex numbering
    // 3) migrate cells (connectivity + shape + GIDs)
    // 4) distribute only needed vertex coordinates by vertex-block ownership
    // 5) build local MeshBase on each rank
#if defined(SVMP_HAS_PARMETIS)
    if (partition_method == "parmetis") {
        build_from_arrays_global_and_partition_two_phase_parmetis_(
            spatial_dim, X_ref, cell2vertex_offsets, cell2vertex, cell_shape, hint, ghost_layers, options);
        return;
    }
#endif

	    // Rank 0 builds the global mesh, partitions by cells, and distributes rank-local submeshes.
	    if (my_rank_ == 0) {
	        MeshBase global_mesh;
	        global_mesh.build_from_arrays(spatial_dim, X_ref, cell2vertex_offsets, cell2vertex, cell_shape);
		        global_mesh.finalize();

	        const size_t n_cells = global_mesh.n_cells();
	        std::vector<rank_t> cell_partition;

            if (partition_method == "parmetis") {
                // ParMETIS startup partitioning is performed after an initial block distribution.
                // The initial distribution only needs to provide a valid mesh per rank.
                cell_partition = partition_cells_block(n_cells, world_size_);
            } else if (partition_method == "block") {
                cell_partition = partition_cells_block(n_cells, world_size_);
            } else {
#if defined(SVMP_HAS_METIS)
                if (partition_method == "metis" || partition_method == "graph") {
                    cell_partition = partition_cells_metis(global_mesh, world_size_, hint);
                } else {
                    cell_partition = partition_cells_metis(global_mesh, world_size_, hint);
                }
#else
                (void)hint;
                cell_partition = partition_cells_block(n_cells, world_size_);
#endif
            }

            // Precompute per-rank cell lists so we don't rescan the global mesh for each rank.
            std::vector<size_t> cell_counts(static_cast<size_t>(world_size_), 0);
            for (size_t c = 0; c < n_cells; ++c) {
                const auto r = cell_partition[c];
                if (r < 0 || r >= world_size_) continue;
                cell_counts[static_cast<size_t>(r)]++;
            }

            std::vector<std::vector<index_t>> cells_by_rank(static_cast<size_t>(world_size_));
            for (rank_t r = 0; r < world_size_; ++r) {
                cells_by_rank[static_cast<size_t>(r)].reserve(cell_counts[static_cast<size_t>(r)]);
            }
            for (index_t c = 0; c < static_cast<index_t>(n_cells); ++c) {
                const auto r = cell_partition[static_cast<size_t>(c)];
                if (r < 0 || r >= world_size_) continue;
                cells_by_rank[static_cast<size_t>(r)].push_back(c);
            }

	        for (rank_t r = 0; r < world_size_; ++r) {
                MeshBase submesh = extract_submesh(global_mesh,
                                                   cell_partition,
                                                   r,
                                                   &cells_by_rank[static_cast<size_t>(r)]);

                if (r == 0) {
                    *local_mesh_ = std::move(submesh);
                    continue;
                }

                std::vector<char> buffer;
                serialize_mesh(submesh, buffer);
                int buffer_size = static_cast<int>(buffer.size());

                MPI_Send(&buffer_size, 1, MPI_INT, r, 200, comm_);
                if (buffer_size > 0) {
                    MPI_Send(buffer.data(), buffer_size, MPI_CHAR, r, 201, comm_);
                }
            }
    } else {
        int buffer_size = 0;
        MPI_Recv(&buffer_size, 1, MPI_INT, 0, 200, comm_, MPI_STATUS_IGNORE);
        if (buffer_size < 0) {
            throw std::runtime_error(
                "DistributedMesh::build_from_arrays_global_and_partition: received invalid buffer size");
        }

        std::vector<char> buffer(static_cast<size_t>(buffer_size));
        if (buffer_size > 0) {
            MPI_Recv(buffer.data(), buffer_size, MPI_CHAR, 0, 201, comm_, MPI_STATUS_IGNORE);
        }
        deserialize_mesh(buffer, *local_mesh_);
    }

    // Establish ownership/shared state and build exchange patterns for ghost/field exchange.
    reset_partition_state_();

#if defined(SVMP_HAS_PARMETIS)
    if (partition_method == "parmetis") {
        // Use the caller's weight hint (Cells/Vertices/Memory) while selecting ParMETIS via partition_method.
        rebalance(hint, options);
    }
#endif
    if (ghost_layers > 0) {
        // build_ghost_layer() re-establishes ownership and rebuilds exchange patterns.
        build_ghost_layer(ghost_layers);
    } else {
        build_exchange_patterns();
    }
#else
    (void)hint;
    (void)ghost_layers;
    (void)options;
    build_from_arrays_local(spatial_dim, X_ref, cell2vertex_offsets, cell2vertex, cell_shape);
#endif
}

// ==========================================
// MPI Info
// ==========================================

DistributedMesh::CommHolder::~CommHolder() {
#ifdef MESH_HAS_MPI
    if (!owns) {
        return;
    }
    if (comm == MPI_COMM_NULL) {
        return;
    }

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        return;
    }
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized) {
        return;
    }

    MPI_Comm_free(&comm);
#endif
}

void DistributedMesh::set_mpi_comm(MPI_Comm comm) {
    user_comm_ = comm;

#ifdef MESH_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    MPI_Comm internal_comm = MPI_COMM_SELF;
    bool owns_internal_comm = false;

    if (initialized && comm != MPI_COMM_NULL) {
      internal_comm = comm;
      if (MPI_Comm_dup(comm, &internal_comm) == MPI_SUCCESS) {
        owns_internal_comm = true;
      } else {
        // Fall back to using the provided communicator directly.
        internal_comm = comm;
        owns_internal_comm = false;
      }

      MPI_Comm_rank(internal_comm, &my_rank_);
      MPI_Comm_size(internal_comm, &world_size_);
    } else {
      my_rank_ = 0;
      world_size_ = 1;
      internal_comm = MPI_COMM_SELF;
      owns_internal_comm = false;
    }
#else
    my_rank_ = 0;
    world_size_ = 1;
#endif

    comm_ = internal_comm;
    comm_holder_ = std::make_shared<CommHolder>();
    comm_holder_->comm = internal_comm;
    comm_holder_->owns = owns_internal_comm;
}

// ==========================================
// Ownership & Ghosting
// ==========================================

bool DistributedMesh::is_owned_cell(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(cell_owner_.size())) {
        return false;
    }
    return cell_owner_[i] == Ownership::Owned;
}

bool DistributedMesh::is_ghost_cell(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(cell_owner_.size())) {
        return false;
    }
    return cell_owner_[i] == Ownership::Ghost;
}

bool DistributedMesh::is_shared_cell(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(cell_owner_.size())) {
        return false;
    }
    return cell_owner_[i] == Ownership::Shared;
}

rank_t DistributedMesh::owner_rank_cell(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(cell_owner_rank_.size())) {
        return -1;
    }
    return cell_owner_rank_[i];
}

bool DistributedMesh::is_owned_vertex(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(vertex_owner_.size())) {
        return false;
    }
    return vertex_owner_[i] == Ownership::Owned;
}

bool DistributedMesh::is_ghost_vertex(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(vertex_owner_.size())) {
        return false;
    }
    return vertex_owner_[i] == Ownership::Ghost;
}

bool DistributedMesh::is_shared_vertex(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(vertex_owner_.size())) {
        return false;
    }
    return vertex_owner_[i] == Ownership::Shared;
}

rank_t DistributedMesh::owner_rank_vertex(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(vertex_owner_rank_.size())) {
        return -1;
    }
    return vertex_owner_rank_[i];
}

bool DistributedMesh::is_owned_face(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(face_owner_.size())) {
        return false;
    }
    return face_owner_[i] == Ownership::Owned;
}

bool DistributedMesh::is_ghost_face(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(face_owner_.size())) {
        return false;
    }
    return face_owner_[i] == Ownership::Ghost;
}

bool DistributedMesh::is_shared_face(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(face_owner_.size())) {
        return false;
    }
    return face_owner_[i] == Ownership::Shared;
}

rank_t DistributedMesh::owner_rank_face(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(face_owner_rank_.size())) {
        return -1;
    }
    return face_owner_rank_[i];
}

bool DistributedMesh::is_owned_edge(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(edge_owner_.size())) {
        return false;
    }
    return edge_owner_[i] == Ownership::Owned;
}

bool DistributedMesh::is_ghost_edge(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(edge_owner_.size())) {
        return false;
    }
    return edge_owner_[i] == Ownership::Ghost;
}

bool DistributedMesh::is_shared_edge(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(edge_owner_.size())) {
        return false;
    }
    return edge_owner_[i] == Ownership::Shared;
}

rank_t DistributedMesh::owner_rank_edge(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(edge_owner_rank_.size())) {
        return -1;
    }
    return edge_owner_rank_[i];
}

void DistributedMesh::set_ownership(index_t entity_id, EntityKind kind,
                                   Ownership ownership, rank_t owner_rank) {
    switch (kind) {
        case EntityKind::Vertex:
            if (entity_id >= 0 && entity_id < static_cast<index_t>(vertex_owner_.size())) {
                vertex_owner_[entity_id] = ownership;
                if (owner_rank >= 0) {
                    vertex_owner_rank_[entity_id] = owner_rank;
                }
            }
            break;
        case EntityKind::Face:
            if (entity_id >= 0 && entity_id < static_cast<index_t>(face_owner_.size())) {
                face_owner_[entity_id] = ownership;
                if (owner_rank >= 0) {
                    face_owner_rank_[entity_id] = owner_rank;
                }
            }
            break;
        case EntityKind::Edge:
            if (entity_id >= 0 && entity_id < static_cast<index_t>(edge_owner_.size())) {
                edge_owner_[entity_id] = ownership;
                if (owner_rank >= 0) {
                    edge_owner_rank_[entity_id] = owner_rank;
                }
            }
            break;
        case EntityKind::Volume:
            if (entity_id >= 0 && entity_id < static_cast<index_t>(cell_owner_.size())) {
                cell_owner_[entity_id] = ownership;
                if (owner_rank >= 0) {
                    cell_owner_rank_[entity_id] = owner_rank;
                }
            }
            break;
        default:
            break;
    }
    if (local_mesh_) {
        local_mesh_->event_bus().notify(MeshEvent::PartitionChanged);
    }
}

// ==========================================
// Ghost Layer Construction
// ==========================================

#ifdef MESH_HAS_MPI
namespace {

struct FaceKey {
    int8_t n = 0;
    std::array<gid_t,4> v{{INVALID_GID, INVALID_GID, INVALID_GID, INVALID_GID}};
};

struct FaceKeyHash {
    size_t operator()(const FaceKey& k) const noexcept {
        uint64_t h = 0xcbf29ce484222325ULL;
        auto mix = [&h](uint64_t x) {
            h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(static_cast<uint64_t>(k.n));
        for (int i = 0; i < k.n; ++i) {
            mix(static_cast<uint64_t>(k.v[static_cast<size_t>(i)]));
        }
        return static_cast<size_t>(h);
    }
};

struct FaceKeyEq {
    bool operator()(const FaceKey& a, const FaceKey& b) const noexcept {
        if (a.n != b.n) return false;
        for (int i = 0; i < a.n; ++i) {
            if (a.v[static_cast<size_t>(i)] != b.v[static_cast<size_t>(i)]) return false;
        }
        return true;
    }
};

struct BoundaryFaceRecord {
    FaceKey key;
    gid_t adjacent_cell_gid = INVALID_GID;
    rank_t src_rank = -1;
};

struct FaceMatchRecord {
    rank_t neighbor_rank = -1;
    gid_t neighbor_cell_gid = INVALID_GID;
};

struct FaceAdjQuery {
    FaceKey key;
    gid_t adjacent_cell_gid = INVALID_GID;
};

struct FaceAdjReply {
    gid_t neighbor_cell_gid = INVALID_GID;
    rank_t neighbor_owner_rank = -1;
};

inline FaceKey make_face_key(const MeshBase& mesh, index_t face_id) {
    FaceKey key;
    auto [verts, n] = mesh.face_vertices_span(face_id);
    if (n <= 0 || n > 4) return key;
    key.n = static_cast<int8_t>(n);
    const auto& vg = mesh.vertex_gids();
    for (size_t i = 0; i < n; ++i) {
        key.v[i] = vg[static_cast<size_t>(verts[i])];
    }
    std::sort(key.v.begin(), key.v.begin() + n);
    return key;
}

// Extract a submesh containing a set of cells specified by global IDs.
inline MeshBase extract_cells_by_gid(const MeshBase& mesh, const std::vector<gid_t>& cell_gids) {
    MeshBase submesh;
    const int dim = mesh.dim();

    std::unordered_map<gid_t, index_t> gid_to_local_vertex;
    gid_to_local_vertex.reserve(cell_gids.size() * 8);

    std::vector<gid_t> new_vertex_gids;
    std::vector<real_t> new_coords;
    std::vector<gid_t> new_cell_gids;
    std::vector<CellShape> new_cell_shapes;
    std::vector<offset_t> new_offsets{0};
    std::vector<index_t> new_conn;

    const auto& X = mesh.X_ref();
    const auto& vg = mesh.vertex_gids();

    for (gid_t cg : cell_gids) {
        const index_t c = mesh.global_to_local_cell(cg);
        if (c == INVALID_INDEX) continue;

        new_cell_gids.push_back(cg);
        new_cell_shapes.push_back(mesh.cell_shape(c));

        auto [cverts, nverts] = mesh.cell_vertices_span(c);
        for (size_t i = 0; i < nverts; ++i) {
            const index_t v = cverts[i];
            const gid_t gid = vg[static_cast<size_t>(v)];
            auto it = gid_to_local_vertex.find(gid);
            if (it == gid_to_local_vertex.end()) {
                const index_t new_id = static_cast<index_t>(new_vertex_gids.size());
                gid_to_local_vertex[gid] = new_id;
                new_vertex_gids.push_back(gid);
                for (int d = 0; d < dim; ++d) {
                    new_coords.push_back(X[static_cast<size_t>(v) * dim + d]);
                }
            }
            new_conn.push_back(gid_to_local_vertex[gid]);
        }
        new_offsets.push_back(static_cast<offset_t>(new_conn.size()));
    }

    // Empty packet: return a default-constructed mesh (caller decides how to handle).
    if (new_cell_gids.empty()) {
        return submesh;
    }

    submesh.build_from_arrays(dim, new_coords, new_offsets, new_conn, new_cell_shapes);
    submesh.set_vertex_gids(std::move(new_vertex_gids));
    submesh.set_cell_gids(std::move(new_cell_gids));
    submesh.finalize();
    return submesh;
}

} // namespace
#endif // MESH_HAS_MPI

void DistributedMesh::build_ghost_layer(int levels) {
#ifdef MESH_HAS_MPI
    if (world_size_ == 1) return;
    if (levels <= 0) {
        clear_ghosts();
        return;
    }

    // Always start from a clean (no-ghost) state.
    if (ghost_levels_ > 0 || !ghost_cells_.empty() || !ghost_vertices_.empty()) {
        clear_ghosts();
    }

    // Ensure ownership arrays exist for the base partition (owned/shared).
    gather_shared_entities();

    ghost_levels_ = levels;

	    auto rebuild_with_ghost_meshes = [&](const std::vector<std::pair<rank_t, MeshBase>>& incoming) -> std::vector<gid_t> {
        // Ghost-layer construction is a collective operation. Even ranks that do not
        // receive any new ghost entities in a given layer must still participate in
        // the shared-entity detection collectives to avoid deadlocks.
        if (incoming.empty()) {
            std::vector<index_t> explicit_ghost_vertices;
            explicit_ghost_vertices.reserve(ghost_vertices_.size());
            for (index_t v = 0; v < static_cast<index_t>(vertex_owner_.size()); ++v) {
                if (vertex_owner_[static_cast<size_t>(v)] == Ownership::Ghost) {
                    explicit_ghost_vertices.push_back(v);
                }
            }
            const std::vector<index_t> explicit_ghost_cells(ghost_cells_.begin(), ghost_cells_.end());
            const std::vector<index_t> explicit_ghost_faces(ghost_faces_.begin(), ghost_faces_.end());
            const std::vector<index_t> explicit_ghost_edges(ghost_edges_.begin(), ghost_edges_.end());

            gather_shared_entities();

            for (index_t v : explicit_ghost_vertices) {
                vertex_owner_[static_cast<size_t>(v)] = Ownership::Ghost;
            }
            for (index_t c : explicit_ghost_cells) {
                cell_owner_[static_cast<size_t>(c)] = Ownership::Ghost;
            }
            for (index_t f : explicit_ghost_faces) {
                face_owner_[static_cast<size_t>(f)] = Ownership::Ghost;
            }
            for (index_t e : explicit_ghost_edges) {
                edge_owner_[static_cast<size_t>(e)] = Ownership::Ghost;
            }

            return {};
        }

	        const size_t old_n_vertices = local_mesh_->n_vertices();
	        const size_t old_n_cells = local_mesh_->n_cells();
	        const size_t old_n_faces = local_mesh_->n_faces();
	        const size_t old_n_edges = local_mesh_->n_edges();
	        const auto old_cell_regions = local_mesh_->cell_region_ids();
	        const auto old_cell_refinement_levels = local_mesh_->cell_refinement_levels();
	        const auto old_vertex_labels = local_mesh_->vertex_label_ids();
	        const auto old_face_boundary_labels = local_mesh_->face_boundary_ids();
	        const auto old_edge_labels = local_mesh_->edge_label_ids();
	        const auto label_registry = local_mesh_->list_label_names();
	        const Configuration old_active_config = local_mesh_->active_configuration();
	        const bool old_has_current_coords = local_mesh_->has_current_coords();
	        std::vector<real_t> old_current_coords;
	        if (old_has_current_coords) {
	            old_current_coords = local_mesh_->X_cur();
	        }

	        const auto old_vertex_owner = vertex_owner_;
	        const auto old_vertex_owner_rank = vertex_owner_rank_;
	        const auto old_cell_owner = cell_owner_;
	        const auto old_cell_owner_rank = cell_owner_rank_;

        // Preserve field data/metadata across ghost-layer rebuilds.
        struct FieldSnapshot {
            std::string name;
            EntityKind kind;
            FieldScalarType type;
            size_t components;
            size_t bytes_per_component;
            bool has_descriptor = false;
            FieldDescriptor descriptor;
            std::shared_ptr<const std::vector<gid_t>> gids; // for face/edge remapping by GID
            std::vector<uint8_t> data;
        };

        std::vector<FieldSnapshot> saved_fields;

        auto snapshot_fields = [&](EntityKind kind,
                                   size_t n_entities,
                                   std::shared_ptr<const std::vector<gid_t>> gids = {}) {
            for (const auto& fname : local_mesh_->field_names(kind)) {
                FieldSnapshot snap;
                snap.name = fname;
                snap.kind = kind;
                snap.type = local_mesh_->field_type_by_name(kind, fname);
                snap.components = local_mesh_->field_components_by_name(kind, fname);
                snap.bytes_per_component = local_mesh_->field_bytes_per_component_by_name(kind, fname);
                snap.gids = gids;

                const auto hnd = local_mesh_->field_handle(kind, fname);
                if (hnd.id != 0) {
                    if (const auto* desc = local_mesh_->field_descriptor(hnd)) {
                        snap.has_descriptor = true;
                        snap.descriptor = *desc;
                    }
                }

                const uint8_t* src = static_cast<const uint8_t*>(local_mesh_->field_data_by_name(kind, fname));
                if (!src) continue;

                const size_t bpe = snap.components * snap.bytes_per_component;
                snap.data.assign(src, src + n_entities * bpe);
                saved_fields.push_back(std::move(snap));
            }
        };

	        // Ensure codim-1/codim-2 GIDs are stable before snapshotting face/edge fields.
	        ensure_canonical_face_gids(*local_mesh_);
	        ensure_canonical_edge_gids(*local_mesh_);
	        auto face_gid_snapshot = std::make_shared<std::vector<gid_t>>(local_mesh_->face_gids());
	        auto edge_gid_snapshot = std::make_shared<std::vector<gid_t>>(local_mesh_->edge_gids());

	        // Preserve entity sets (Vertex/Volume by index; Face/Edge by canonical GID).
	        struct SetSnapshot {
	            EntityKind kind = EntityKind::Vertex;
	            std::string name;
	            std::vector<index_t> ids;
	            std::vector<gid_t> gids;
	        };
	        std::vector<SetSnapshot> saved_sets;

	        auto snapshot_index_sets = [&](EntityKind kind, size_t max_id) {
	            for (const auto& set_name : local_mesh_->list_sets(kind)) {
	                const auto& ids = local_mesh_->get_set(kind, set_name);
	                std::vector<index_t> kept;
	                kept.reserve(ids.size());
	                for (const auto id : ids) {
	                    if (id < 0 || static_cast<size_t>(id) >= max_id) continue;
	                    kept.push_back(id);
	                }
	                if (!kept.empty()) {
	                    saved_sets.push_back({kind, set_name, std::move(kept), {}});
	                }
	            }
	        };

	        auto snapshot_gid_sets = [&](EntityKind kind, const std::vector<gid_t>& gids) {
	            for (const auto& set_name : local_mesh_->list_sets(kind)) {
	                const auto& ids = local_mesh_->get_set(kind, set_name);
	                std::vector<gid_t> members;
	                members.reserve(ids.size());
	                for (const auto id : ids) {
	                    if (id < 0 || static_cast<size_t>(id) >= gids.size()) continue;
	                    const gid_t gid = gids[static_cast<size_t>(id)];
	                    if (gid == INVALID_GID) continue;
	                    members.push_back(gid);
	                }
	                if (!members.empty()) {
	                    saved_sets.push_back({kind, set_name, {}, std::move(members)});
	                }
	            }
	        };

	        snapshot_index_sets(EntityKind::Vertex, old_n_vertices);
	        snapshot_index_sets(EntityKind::Volume, old_n_cells);
	        snapshot_gid_sets(EntityKind::Face, *face_gid_snapshot);
	        snapshot_gid_sets(EntityKind::Edge, *edge_gid_snapshot);

	        snapshot_fields(EntityKind::Vertex, old_n_vertices);
	        snapshot_fields(EntityKind::Volume, old_n_cells);
	        snapshot_fields(EntityKind::Face, old_n_faces, face_gid_snapshot);
	        snapshot_fields(EntityKind::Edge, old_n_edges, edge_gid_snapshot);

        // Start with current mesh arrays.
        std::vector<real_t> new_coords = local_mesh_->X_ref();
        std::vector<gid_t> new_vertex_gids = local_mesh_->vertex_gids();

        std::unordered_map<gid_t, index_t> gid_to_new_vertex;
        gid_to_new_vertex.reserve(new_vertex_gids.size() + 1024);
        for (index_t v = 0; v < static_cast<index_t>(new_vertex_gids.size()); ++v) {
            gid_to_new_vertex[new_vertex_gids[static_cast<size_t>(v)]] = v;
        }

        // Track owner ranks for newly created vertices (best-effort).
        std::unordered_map<gid_t, rank_t> new_vertex_owner;

        // Add new vertices from incoming meshes.
        for (const auto& [owner_rank, m] : incoming) {
            const auto& vg = m.vertex_gids();
            const auto& X = m.X_ref();
            const int dim = m.dim();

            for (index_t v = 0; v < static_cast<index_t>(vg.size()); ++v) {
                const gid_t gid = vg[static_cast<size_t>(v)];
                if (gid_to_new_vertex.find(gid) != gid_to_new_vertex.end()) continue;
                const index_t new_id = static_cast<index_t>(new_vertex_gids.size());
                gid_to_new_vertex[gid] = new_id;
                new_vertex_gids.push_back(gid);
                for (int d = 0; d < dim; ++d) {
                    new_coords.push_back(X[static_cast<size_t>(v) * dim + d]);
                }
                new_vertex_owner[gid] = owner_rank;
            }
        }

        std::vector<CellShape> new_cell_shapes = local_mesh_->cell_shapes();
        std::vector<offset_t> new_offsets = local_mesh_->cell2vertex_offsets();
        std::vector<index_t> new_conn = local_mesh_->cell2vertex();
        std::vector<gid_t> new_cell_gids = local_mesh_->cell_gids();

        std::unordered_set<gid_t> present_cells;
        present_cells.reserve(new_cell_gids.size() + 1024);
        for (gid_t gid : new_cell_gids) present_cells.insert(gid);

        std::vector<std::pair<gid_t, rank_t>> appended_cells; // (cell_gid, owner_rank)

        // Append incoming ghost cells.
        for (const auto& [owner_rank, m] : incoming) {
            const auto& cell_gids = m.cell_gids();
            const auto& vg = m.vertex_gids();

            for (index_t c = 0; c < static_cast<index_t>(cell_gids.size()); ++c) {
                const gid_t cg = cell_gids[static_cast<size_t>(c)];
                if (!present_cells.insert(cg).second) continue;

                appended_cells.emplace_back(cg, owner_rank);

                new_cell_gids.push_back(cg);
                new_cell_shapes.push_back(m.cell_shape(c));

                auto [cverts, nverts] = m.cell_vertices_span(c);
                for (size_t i = 0; i < nverts; ++i) {
                    const gid_t vgid = vg[static_cast<size_t>(cverts[i])];
                    new_conn.push_back(gid_to_new_vertex.at(vgid));
                }
                new_offsets.push_back(static_cast<offset_t>(new_conn.size()));
            }
        }

        // Rebuild mesh (this clears fields, so restore after).
        const int dim = local_mesh_->dim();
        local_mesh_->clear();
        local_mesh_->build_from_arrays(dim, new_coords, new_offsets, new_conn, new_cell_shapes);
        local_mesh_->set_vertex_gids(std::move(new_vertex_gids));
	        local_mesh_->set_cell_gids(std::move(new_cell_gids));
	        local_mesh_->finalize();

	        ensure_canonical_face_gids(*local_mesh_);
	        ensure_canonical_edge_gids(*local_mesh_);

	        // Restore communicator-wide label registry (name <-> id).
	        for (const auto& [label, name] : label_registry) {
	            local_mesh_->register_label(name, label);
	        }

	        // Restore current coordinates and active configuration.
	        if (old_has_current_coords) {
	            std::vector<real_t> new_current_coords;
	            new_current_coords.reserve(new_coords.size());
	            new_current_coords.insert(new_current_coords.end(), old_current_coords.begin(), old_current_coords.end());
	            if (new_coords.size() > old_current_coords.size()) {
	                new_current_coords.insert(new_current_coords.end(),
	                                          new_coords.begin() + old_current_coords.size(),
	                                          new_coords.end());
	            }
	            local_mesh_->set_current_coords(new_current_coords);
	        }
	        if (old_active_config == Configuration::Current && old_has_current_coords) {
	            local_mesh_->use_current_configuration();
	        } else {
	            local_mesh_->use_reference_configuration();
	        }

	        // Restore vertex labels for the base partition vertices (indices preserved).
	        if (!old_vertex_labels.empty()) {
	            const size_t n_copy = std::min(old_vertex_labels.size(), old_n_vertices);
	            for (size_t v = 0; v < n_copy; ++v) {
	                const label_t label = old_vertex_labels[v];
	                if (label == INVALID_LABEL) continue;
	                local_mesh_->set_vertex_label(static_cast<index_t>(v), label);
	            }
	        }

	        // Preserve region labels for existing (non-ghost) cells.
	        if (!old_cell_regions.empty()) {
	            const index_t n_keep = static_cast<index_t>(std::min(old_n_cells, local_mesh_->n_cells()));
	            for (index_t c = 0; c < n_keep; ++c) {
	                local_mesh_->set_region_label(c, old_cell_regions[static_cast<size_t>(c)]);
	            }
	        }

	        // Preserve cell refinement levels for existing cells.
	        if (!old_cell_refinement_levels.empty() && old_cell_refinement_levels.size() == old_n_cells) {
	            std::vector<size_t> levels(local_mesh_->n_cells(), 0);
	            const size_t n_keep = std::min(old_n_cells, local_mesh_->n_cells());
	            for (size_t c = 0; c < n_keep; ++c) {
	                levels[c] = old_cell_refinement_levels[c];
	            }
	            local_mesh_->set_cell_refinement_levels(std::move(levels));
	        }

	        // Preserve face boundary labels / edge labels by canonical GID.
	        if (!old_face_boundary_labels.empty() && face_gid_snapshot && !face_gid_snapshot->empty()) {
	            const size_t n = std::min(old_face_boundary_labels.size(), face_gid_snapshot->size());
	            for (size_t i = 0; i < n; ++i) {
	                const label_t label = old_face_boundary_labels[i];
	                if (label == INVALID_LABEL) continue;
	                const gid_t gid = (*face_gid_snapshot)[i];
	                if (gid == INVALID_GID) continue;
	                const index_t f = local_mesh_->global_to_local_face(gid);
	                if (f == INVALID_INDEX) continue;
	                local_mesh_->set_boundary_label(f, label);
	            }
	        }

	        if (!old_edge_labels.empty() && edge_gid_snapshot && !edge_gid_snapshot->empty()) {
	            const size_t n = std::min(old_edge_labels.size(), edge_gid_snapshot->size());
	            for (size_t i = 0; i < n; ++i) {
	                const label_t label = old_edge_labels[i];
	                if (label == INVALID_LABEL) continue;
	                const gid_t gid = (*edge_gid_snapshot)[i];
	                if (gid == INVALID_GID) continue;
	                const index_t e = local_mesh_->global_to_local_edge(gid);
	                if (e == INVALID_INDEX) continue;
	                local_mesh_->set_edge_label(e, label);
	            }
	        }

	        // Restore entity sets.
	        for (const auto& set : saved_sets) {
	            if (set.kind == EntityKind::Vertex || set.kind == EntityKind::Volume) {
	                for (const auto id : set.ids) {
	                    local_mesh_->add_to_set(set.kind, set.name, id);
	                }
	            } else if (set.kind == EntityKind::Face) {
	                for (const auto gid : set.gids) {
	                    const index_t f = local_mesh_->global_to_local_face(gid);
	                    if (f == INVALID_INDEX) continue;
	                    local_mesh_->add_to_set(EntityKind::Face, set.name, f);
	                }
	            } else if (set.kind == EntityKind::Edge) {
	                for (const auto gid : set.gids) {
	                    const index_t e = local_mesh_->global_to_local_edge(gid);
	                    if (e == INVALID_INDEX) continue;
	                    local_mesh_->add_to_set(EntityKind::Edge, set.name, e);
	                }
	            }
	        }

	        // Restore fields (all kinds), remapping face/edge data by canonical GIDs.
	        for (const auto& snap : saved_fields) {
	            const size_t bpe = snap.components * snap.bytes_per_component;
	            auto h = local_mesh_->attach_field(snap.kind, snap.name, snap.type, snap.components, snap.bytes_per_component);
	            uint8_t* dst = static_cast<uint8_t*>(local_mesh_->field_data(h));
            if (!dst) continue;

            const size_t new_count = local_mesh_->field_entity_count(h);
            std::memset(dst, 0, new_count * bpe);

            if (snap.kind == EntityKind::Vertex || snap.kind == EntityKind::Volume) {
                const size_t n_copy = std::min(snap.data.size(), new_count * bpe);
                std::memcpy(dst, snap.data.data(), n_copy);
            } else if (snap.kind == EntityKind::Face || snap.kind == EntityKind::Edge) {
                if (!snap.gids) continue;
                const auto& gids = *snap.gids;
                const size_t n_old = std::min(gids.size(), snap.data.size() / bpe);
                for (size_t i = 0; i < n_old; ++i) {
                    const gid_t gid = gids[i];
                    const index_t new_id =
                        (snap.kind == EntityKind::Face)
                            ? local_mesh_->global_to_local_face(gid)
                            : local_mesh_->global_to_local_edge(gid);
                    if (new_id == INVALID_INDEX || new_id < 0 || static_cast<size_t>(new_id) >= new_count) {
                        continue;
                    }
                    std::memcpy(dst + static_cast<size_t>(new_id) * bpe,
                                snap.data.data() + i * bpe,
                                bpe);
                }
            }

            if (snap.has_descriptor) {
                local_mesh_->set_field_descriptor(h, snap.descriptor);
            }
        }

        // Resize ownership arrays.
        vertex_owner_.resize(local_mesh_->n_vertices(), Ownership::Owned);
        vertex_owner_rank_.resize(local_mesh_->n_vertices(), my_rank_);
        cell_owner_.resize(local_mesh_->n_cells(), Ownership::Owned);
        cell_owner_rank_.resize(local_mesh_->n_cells(), my_rank_);

        // Preserve old ownership for existing entities.
        for (size_t i = 0; i < std::min(old_n_vertices, vertex_owner_.size()); ++i) {
            vertex_owner_[i] = old_vertex_owner[i];
            vertex_owner_rank_[i] = old_vertex_owner_rank[i];
        }
        for (size_t i = 0; i < std::min(old_n_cells, cell_owner_.size()); ++i) {
            cell_owner_[i] = old_cell_owner[i];
            cell_owner_rank_[i] = old_cell_owner_rank[i];
        }

        // Mark newly appended vertices as ghosts (best-effort owner rank).
        const auto& vg_final = local_mesh_->vertex_gids();
        for (index_t v = static_cast<index_t>(old_n_vertices); v < static_cast<index_t>(local_mesh_->n_vertices()); ++v) {
            vertex_owner_[v] = Ownership::Ghost;
            auto it = new_vertex_owner.find(vg_final[static_cast<size_t>(v)]);
            vertex_owner_rank_[v] = (it != new_vertex_owner.end()) ? it->second : -1;
        }

        // Mark newly appended cells as ghosts with the sending owner rank.
        std::vector<gid_t> newly_added_cell_gids;
        newly_added_cell_gids.reserve(appended_cells.size());

        const auto& cg_final = local_mesh_->cell_gids();
        for (index_t c = static_cast<index_t>(old_n_cells); c < static_cast<index_t>(local_mesh_->n_cells()); ++c) {
            cell_owner_[c] = Ownership::Ghost;
            const gid_t gid = cg_final[static_cast<size_t>(c)];
            // appended_cells is in the same order as appended to new_cell_gids.
            const size_t appended_idx = static_cast<size_t>(c) - old_n_cells;
            if (appended_idx < appended_cells.size()) {
                cell_owner_rank_[c] = appended_cells[appended_idx].second;
            } else {
                cell_owner_rank_[c] = -1;
            }
            newly_added_cell_gids.push_back(gid);
        }

        // Faces are rebuilt by MeshBase::finalize(); set a conservative ownership mark.
        face_owner_.assign(local_mesh_->n_faces(), Ownership::Owned);
        face_owner_rank_.assign(local_mesh_->n_faces(), my_rank_);
        for (index_t f = 0; f < static_cast<index_t>(local_mesh_->n_faces()); ++f) {
            auto fc = local_mesh_->face_cells(f);
            const index_t c0 = fc[0];
            const index_t c1 = fc[1];
            const bool c0_valid = (c0 != INVALID_INDEX && c0 < static_cast<index_t>(cell_owner_.size()));
            const bool c1_valid = (c1 != INVALID_INDEX && c1 < static_cast<index_t>(cell_owner_.size()));
            const bool c0_ghost = c0_valid && cell_owner_[c0] == Ownership::Ghost;
            const bool c1_ghost = c1_valid && cell_owner_[c1] == Ownership::Ghost;
            const bool c0_non_ghost = c0_valid && cell_owner_[c0] != Ownership::Ghost;
            const bool c1_non_ghost = c1_valid && cell_owner_[c1] != Ownership::Ghost;

            // A face is considered ghost only when *all* incident cells are ghost.
            // Faces on the owned/ghost interface remain owned/shared so they contribute
            // to deterministic ownership and global counts.
            const bool has_ghost = c0_ghost || c1_ghost;
            const bool has_non_ghost = c0_non_ghost || c1_non_ghost;
            if (has_ghost && !has_non_ghost) {
                face_owner_[f] = Ownership::Ghost;
                const index_t gc = c0_ghost ? c0 : c1;
                face_owner_rank_[f] = cell_owner_rank_[gc];
            }
        }

        // Edges are rebuilt by MeshBase::finalize(); set a conservative ownership mark.
        edge_owner_.assign(local_mesh_->n_edges(), Ownership::Owned);
        edge_owner_rank_.assign(local_mesh_->n_edges(), my_rank_);
        if (local_mesh_->n_edges() > 0 && !local_mesh_->edge2vertex().empty()) {
            std::vector<offset_t> edge2cell_offsets;
            std::vector<index_t> edge2cell;
            MeshTopology::build_edge2cell(*local_mesh_, local_mesh_->edge2vertex(), edge2cell_offsets, edge2cell);

            for (index_t e = 0; e < static_cast<index_t>(local_mesh_->n_edges()); ++e) {
                const offset_t start = edge2cell_offsets[static_cast<size_t>(e)];
                const offset_t end = edge2cell_offsets[static_cast<size_t>(e + 1)];
                if (start >= end) continue;

                bool has_ghost = false;
                bool has_non_ghost = false;
                rank_t min_ghost_owner = -1;

                for (offset_t off = start; off < end; ++off) {
                    const index_t c = edge2cell[static_cast<size_t>(off)];
                    if (c < 0 || static_cast<size_t>(c) >= cell_owner_.size()) continue;
                    if (cell_owner_[static_cast<size_t>(c)] == Ownership::Ghost) {
                        has_ghost = true;
                        const rank_t r = cell_owner_rank_[static_cast<size_t>(c)];
                        if (r >= 0) {
                            min_ghost_owner = (min_ghost_owner < 0) ? r : std::min(min_ghost_owner, r);
                        }
                    } else {
                        has_non_ghost = true;
                    }
                }

                if (has_ghost && !has_non_ghost) {
                    edge_owner_[static_cast<size_t>(e)] = Ownership::Ghost;
                    edge_owner_rank_[static_cast<size_t>(e)] = min_ghost_owner;
                }
            }
        }

        // Update ghost sets.
        ghost_cells_.clear();
        ghost_vertices_.clear();
        ghost_faces_.clear();
        ghost_edges_.clear();

        for (index_t c = 0; c < static_cast<index_t>(cell_owner_.size()); ++c) {
            if (cell_owner_[c] == Ownership::Ghost) {
                ghost_cells_.insert(c);
            }
        }

        for (index_t f = 0; f < static_cast<index_t>(face_owner_.size()); ++f) {
            if (face_owner_[f] == Ownership::Ghost) {
                ghost_faces_.insert(f);
            }
        }

        for (index_t e = 0; e < static_cast<index_t>(edge_owner_.size()); ++e) {
            if (edge_owner_[static_cast<size_t>(e)] == Ownership::Ghost) {
                ghost_edges_.insert(e);
            }
        }

        for (index_t c : ghost_cells_) {
            auto [cverts, nverts] = local_mesh_->cell_vertices_span(c);
            for (size_t i = 0; i < nverts; ++i) {
                ghost_vertices_.insert(cverts[i]);
            }
        }

        // Recompute canonical owner ranks for all entities (including imported ghosts)
        // based on global IDs so ghost/shared field updates can fetch directly from owners.
        //
        // Without this, newly imported vertices may inherit the sending cell-rank as their
        // owner (even when the canonical owner is a different rank), which can require
        // multi-hop propagation and breaks single-step ghost updates.
        std::vector<index_t> explicit_ghost_vertices;
        explicit_ghost_vertices.reserve(ghost_vertices_.size());
        for (index_t v = 0; v < static_cast<index_t>(vertex_owner_.size()); ++v) {
            if (vertex_owner_[static_cast<size_t>(v)] == Ownership::Ghost) {
                explicit_ghost_vertices.push_back(v);
            }
        }
        const std::vector<index_t> explicit_ghost_cells(ghost_cells_.begin(), ghost_cells_.end());
        const std::vector<index_t> explicit_ghost_faces(ghost_faces_.begin(), ghost_faces_.end());
        const std::vector<index_t> explicit_ghost_edges(ghost_edges_.begin(), ghost_edges_.end());

        gather_shared_entities();

        for (index_t v : explicit_ghost_vertices) {
            vertex_owner_[static_cast<size_t>(v)] = Ownership::Ghost;
        }
        for (index_t c : explicit_ghost_cells) {
            cell_owner_[static_cast<size_t>(c)] = Ownership::Ghost;
        }
        for (index_t f : explicit_ghost_faces) {
            face_owner_[static_cast<size_t>(f)] = Ownership::Ghost;
        }
        for (index_t e : explicit_ghost_edges) {
            edge_owner_[static_cast<size_t>(e)] = Ownership::Ghost;
        }

        return newly_added_cell_gids;
    };

    auto alltoallv_records = [&](const std::vector<char>& send_buffer,
                                 const std::vector<int>& send_counts,
                                 std::vector<char>& recv_buffer,
                                 std::vector<int>& recv_counts,
                                 std::vector<int>& recv_displs) {
        std::vector<int> send_displs(world_size_ + 1, 0);
        for (int r = 0; r < world_size_; ++r) {
            send_displs[r + 1] = send_displs[r] + send_counts[r];
        }

        recv_counts.assign(world_size_, 0);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm_);

        recv_displs.assign(world_size_ + 1, 0);
        for (int r = 0; r < world_size_; ++r) {
            recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
        }

        recv_buffer.assign(static_cast<size_t>(recv_displs[world_size_]), 0);

        MPI_Alltoallv(send_buffer.data(),
                      send_counts.data(),
                      send_displs.data(),
                      MPI_BYTE,
                      recv_buffer.data(),
                      recv_counts.data(),
                      recv_displs.data(),
                      MPI_BYTE,
                      comm_);
    };

    auto match_partition_faces = [&]() -> std::unordered_map<rank_t, std::unordered_set<gid_t>> {
        // Send all boundary faces (keyed by vertex GIDs) to their home ranks, match pairs, and
        // return the set of remote cell gids we need to import from each neighbor rank.
        std::vector<BoundaryFaceRecord> local_faces;
        const auto boundary = local_mesh_->boundary_faces();
        local_faces.reserve(boundary.size());

        for (index_t f : boundary) {
            const auto fc = local_mesh_->face_cells(f);
            const index_t adj = (fc[0] != INVALID_INDEX) ? fc[0] : fc[1];
            if (adj == INVALID_INDEX) continue;

            // Only consider faces adjacent to locally-owned cells at layer 1.
            if (adj >= static_cast<index_t>(cell_owner_rank_.size()) || cell_owner_rank_[adj] != my_rank_) {
                continue;
            }

            BoundaryFaceRecord rec;
            rec.key = make_face_key(*local_mesh_, f);
            if (rec.key.n == 0) continue;
            rec.adjacent_cell_gid = local_mesh_->cell_gids()[static_cast<size_t>(adj)];
            rec.src_rank = my_rank_;
            local_faces.push_back(rec);
        }

        // Bucket by home rank.
        std::vector<int> send_counts(world_size_, 0);
        std::vector<std::vector<BoundaryFaceRecord>> by_dest(world_size_);

        FaceKeyHash hasher;
        for (const auto& rec : local_faces) {
            const int home = static_cast<int>(hasher(rec.key) % static_cast<size_t>(world_size_));
            by_dest[home].push_back(rec);
        }

        for (int r = 0; r < world_size_; ++r) {
            send_counts[r] = static_cast<int>(by_dest[r].size() * sizeof(BoundaryFaceRecord));
        }

        std::vector<char> send_buffer;
        send_buffer.resize(static_cast<size_t>(std::accumulate(send_counts.begin(), send_counts.end(), 0)));

        {
            size_t offset = 0;
            for (int r = 0; r < world_size_; ++r) {
                const auto bytes = by_dest[r].size() * sizeof(BoundaryFaceRecord);
                if (bytes > 0) {
                    std::memcpy(send_buffer.data() + offset, by_dest[r].data(), bytes);
                    offset += bytes;
                }
            }
        }

        std::vector<char> recv_buffer;
        std::vector<int> recv_counts, recv_displs;
        alltoallv_records(send_buffer, send_counts, recv_buffer, recv_counts, recv_displs);

        // Group received faces by key.
        std::unordered_map<FaceKey, std::vector<BoundaryFaceRecord>, FaceKeyHash, FaceKeyEq> groups;
        for (int r = 0; r < world_size_; ++r) {
            const int n_bytes = recv_counts[r];
            const int n_recs = n_bytes / static_cast<int>(sizeof(BoundaryFaceRecord));
            const auto* recs = reinterpret_cast<const BoundaryFaceRecord*>(recv_buffer.data() + recv_displs[r]);
            for (int i = 0; i < n_recs; ++i) {
                groups[recs[i].key].push_back(recs[i]);
            }
        }

        // Build match messages to return.
        std::vector<std::vector<FaceMatchRecord>> matches_by_rank(world_size_);
        for (const auto& [key, vec] : groups) {
            if (vec.size() != 2) continue;
            const auto& a = vec[0];
            const auto& b = vec[1];
            if (a.src_rank == b.src_rank) continue;
            matches_by_rank[a.src_rank].push_back({b.src_rank, b.adjacent_cell_gid});
            matches_by_rank[b.src_rank].push_back({a.src_rank, a.adjacent_cell_gid});
        }

        send_counts.assign(world_size_, 0);
        send_buffer.clear();

        for (int r = 0; r < world_size_; ++r) {
            send_counts[r] = static_cast<int>(matches_by_rank[r].size() * sizeof(FaceMatchRecord));
        }

        send_buffer.resize(static_cast<size_t>(std::accumulate(send_counts.begin(), send_counts.end(), 0)));
        {
            size_t offset = 0;
            for (int r = 0; r < world_size_; ++r) {
                const auto bytes = matches_by_rank[r].size() * sizeof(FaceMatchRecord);
                if (bytes > 0) {
                    std::memcpy(send_buffer.data() + offset, matches_by_rank[r].data(), bytes);
                    offset += bytes;
                }
            }
        }

        recv_buffer.clear();
        alltoallv_records(send_buffer, send_counts, recv_buffer, recv_counts, recv_displs);

        std::unordered_map<rank_t, std::unordered_set<gid_t>> requests;
        for (int r = 0; r < world_size_; ++r) {
            const int n_bytes = recv_counts[r];
            const int n_recs = n_bytes / static_cast<int>(sizeof(FaceMatchRecord));
            const auto* recs = reinterpret_cast<const FaceMatchRecord*>(recv_buffer.data() + recv_displs[r]);
            for (int i = 0; i < n_recs; ++i) {
                if (recs[i].neighbor_rank >= 0 && recs[i].neighbor_rank < world_size_) {
                    requests[recs[i].neighbor_rank].insert(recs[i].neighbor_cell_gid);
                }
            }
        }
        return requests;
    };

    auto exchange_cells = [&](const std::unordered_map<rank_t, std::unordered_set<gid_t>>& request_sets) -> std::vector<gid_t> {
        // Phase 1: All-to-all exchange of requested cell IDs.
        std::vector<int> send_counts(world_size_, 0);
        std::vector<int> send_displs(world_size_ + 1, 0);
        std::vector<gid_t> send_gids;

        for (int r = 0; r < world_size_; ++r) {
            auto it = request_sets.find(r);
            if (it == request_sets.end()) continue;
            send_counts[r] = static_cast<int>(it->second.size());
        }

        for (int r = 0; r < world_size_; ++r) {
            send_displs[r + 1] = send_displs[r] + send_counts[r];
        }
        send_gids.assign(static_cast<size_t>(send_displs[world_size_]), INVALID_GID);

        for (int r = 0; r < world_size_; ++r) {
            auto it = request_sets.find(r);
            if (it == request_sets.end()) continue;
            int pos = send_displs[r];
            for (gid_t gid : it->second) {
                send_gids[static_cast<size_t>(pos++)] = gid;
            }
        }

        std::vector<int> recv_counts(world_size_, 0), recv_displs(world_size_ + 1, 0);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm_);
        for (int r = 0; r < world_size_; ++r) {
            recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
        }

        std::vector<gid_t> recv_gids(static_cast<size_t>(recv_displs[world_size_]), INVALID_GID);
        MPI_Alltoallv(send_gids.data(), send_counts.data(), send_displs.data(), MPI_LONG_LONG,
                      recv_gids.data(), recv_counts.data(), recv_displs.data(), MPI_LONG_LONG,
                      comm_);

        // Phase 2: Pack outgoing meshes for ranks that requested from us.
        std::vector<std::vector<char>> outgoing(world_size_);
        for (int r = 0; r < world_size_; ++r) {
            if (recv_counts[r] == 0) continue;
            std::vector<gid_t> req;
            req.reserve(static_cast<size_t>(recv_counts[r]));
            for (int i = 0; i < recv_counts[r]; ++i) {
                req.push_back(recv_gids[static_cast<size_t>(recv_displs[r] + i)]);
            }

            MeshBase sub = extract_cells_by_gid(*local_mesh_, req);
            if (sub.n_cells() > 0) {
                serialize_mesh(sub, outgoing[r]);
            } else {
                outgoing[r].clear();
            }
        }

        // Exchange sizes first.
        const int tag_size = 5000;
        const int tag_data = 5001;

        std::vector<int> incoming_sizes(world_size_, 0);
        std::vector<MPI_Request> reqs;
        reqs.reserve(world_size_ * 2);

        for (int r = 0; r < world_size_; ++r) {
            if (send_counts[r] > 0) {
                MPI_Request q;
                MPI_Irecv(&incoming_sizes[r], 1, MPI_INT, r, tag_size, comm_, &q);
                reqs.push_back(q);
            }
            if (recv_counts[r] > 0) {
                MPI_Request q;
                int sz = static_cast<int>(outgoing[r].size());
                MPI_Isend(&sz, 1, MPI_INT, r, tag_size, comm_, &q);
                reqs.push_back(q);
            }
        }

        if (!reqs.empty()) {
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        }

        // Exchange payloads.
        std::vector<std::vector<char>> incoming(world_size_);
        reqs.clear();

        for (int r = 0; r < world_size_; ++r) {
            if (send_counts[r] > 0) {
                incoming[r].resize(static_cast<size_t>(incoming_sizes[r]));
                MPI_Request q;
                MPI_Irecv(incoming[r].data(), incoming_sizes[r], MPI_CHAR, r, tag_data, comm_, &q);
                reqs.push_back(q);
            }
            if (recv_counts[r] > 0) {
                MPI_Request q;
                MPI_Isend(outgoing[r].data(),
                          static_cast<int>(outgoing[r].size()),
                          MPI_CHAR, r, tag_data, comm_, &q);
                reqs.push_back(q);
            }
        }

        if (!reqs.empty()) {
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        }

        std::vector<std::pair<rank_t, MeshBase>> received_meshes;
        for (int r = 0; r < world_size_; ++r) {
            if (send_counts[r] == 0) continue;
            if (incoming_sizes[r] == 0) continue;
            MeshBase m;
            deserialize_mesh(incoming[r], m);
            if (m.n_cells() > 0) {
                received_meshes.emplace_back(static_cast<rank_t>(r), std::move(m));
            }
        }

        return rebuild_with_ghost_meshes(received_meshes);
    };

    // ---- Layer 1: match partition boundary faces and import adjacent cells.
    auto layer1_requests = match_partition_faces();
    std::vector<gid_t> frontier = exchange_cells(layer1_requests);

    // ---- Additional layers: expand from the last imported ghost cells.
    for (int layer = 2; layer <= levels; ++layer) {
        // All ranks must agree on loop termination to keep MPI collectives aligned.
        int local_has_frontier = frontier.empty() ? 0 : 1;
        int any_has_frontier = 0;
        MPI_Allreduce(&local_has_frontier, &any_has_frontier, 1, MPI_INT, MPI_MAX, comm_);
        if (any_has_frontier == 0) {
            break;
        }

        std::unordered_set<gid_t> frontier_set(frontier.begin(), frontier.end());

        // Build face-key queries for boundary faces adjacent to frontier ghost cells.
        std::vector<std::vector<FaceAdjQuery>> queries_by_rank(world_size_);
        for (index_t f : local_mesh_->boundary_faces()) {
            const auto fc = local_mesh_->face_cells(f);
            const index_t adj = (fc[0] != INVALID_INDEX) ? fc[0] : fc[1];
            if (adj == INVALID_INDEX) continue;
            const gid_t adj_gid = local_mesh_->cell_gids()[static_cast<size_t>(adj)];
            if (frontier_set.find(adj_gid) == frontier_set.end()) continue;

            const rank_t owner = cell_owner_rank_[static_cast<size_t>(adj)];
            if (owner < 0 || owner == my_rank_) continue;

            FaceAdjQuery q;
            q.key = make_face_key(*local_mesh_, f);
            if (q.key.n == 0) continue;
            q.adjacent_cell_gid = adj_gid;
            queries_by_rank[owner].push_back(q);
        }

        // Exchange queries via all-to-allv.
        std::vector<int> send_counts(world_size_, 0);
        for (int r = 0; r < world_size_; ++r) {
            send_counts[r] = static_cast<int>(queries_by_rank[r].size() * sizeof(FaceAdjQuery));
        }

        std::vector<char> send_buffer;
        send_buffer.resize(static_cast<size_t>(std::accumulate(send_counts.begin(), send_counts.end(), 0)));
        {
            size_t off = 0;
            for (int r = 0; r < world_size_; ++r) {
                const auto bytes = queries_by_rank[r].size() * sizeof(FaceAdjQuery);
                if (bytes > 0) {
                    std::memcpy(send_buffer.data() + off, queries_by_rank[r].data(), bytes);
                    off += bytes;
                }
            }
        }

        std::vector<char> recv_buffer;
        std::vector<int> recv_counts, recv_displs;
        alltoallv_records(send_buffer, send_counts, recv_buffer, recv_counts, recv_displs);

        // Build a local lookup table from face key -> face id for answering queries.
        std::unordered_map<FaceKey, index_t, FaceKeyHash, FaceKeyEq> face_lookup;
        face_lookup.reserve(local_mesh_->n_faces() * 2);
        for (index_t f = 0; f < static_cast<index_t>(local_mesh_->n_faces()); ++f) {
            const auto key = make_face_key(*local_mesh_, f);
            if (key.n == 0) continue;
            face_lookup.emplace(key, f);
        }

        // Prepare replies for each source rank, preserving query order.
        std::vector<std::vector<FaceAdjReply>> replies_by_rank(world_size_);
        for (int r = 0; r < world_size_; ++r) {
            const int n_bytes = recv_counts[r];
            const int n_recs = n_bytes / static_cast<int>(sizeof(FaceAdjQuery));
            const auto* qs = reinterpret_cast<const FaceAdjQuery*>(recv_buffer.data() + recv_displs[r]);
            auto& out = replies_by_rank[r];
            out.resize(static_cast<size_t>(n_recs));

            for (int i = 0; i < n_recs; ++i) {
                FaceAdjReply rep;
                auto it = face_lookup.find(qs[i].key);
                if (it != face_lookup.end()) {
                    const index_t f_id = it->second;
                    const auto fc = local_mesh_->face_cells(f_id);
                    const index_t c0 = fc[0];
                    const index_t c1 = fc[1];
                    const gid_t g0 = (c0 != INVALID_INDEX) ? local_mesh_->cell_gids()[static_cast<size_t>(c0)] : INVALID_GID;
                    const gid_t g1 = (c1 != INVALID_INDEX) ? local_mesh_->cell_gids()[static_cast<size_t>(c1)] : INVALID_GID;
                    index_t neigh = INVALID_INDEX;
                    if (c0 != INVALID_INDEX && g0 != qs[i].adjacent_cell_gid) neigh = c0;
                    if (c1 != INVALID_INDEX && g1 != qs[i].adjacent_cell_gid) neigh = c1;

                    if (neigh != INVALID_INDEX) {
                        rep.neighbor_cell_gid = local_mesh_->cell_gids()[static_cast<size_t>(neigh)];
                        rep.neighbor_owner_rank = cell_owner_rank_[static_cast<size_t>(neigh)];
                    }
                }
                out[static_cast<size_t>(i)] = rep;
            }
        }

        // All-to-allv replies back.
        send_counts.assign(world_size_, 0);
        for (int r = 0; r < world_size_; ++r) {
            send_counts[r] = static_cast<int>(replies_by_rank[r].size() * sizeof(FaceAdjReply));
        }

        send_buffer.clear();
        send_buffer.resize(static_cast<size_t>(std::accumulate(send_counts.begin(), send_counts.end(), 0)));
        {
            size_t off = 0;
            for (int r = 0; r < world_size_; ++r) {
                const auto bytes = replies_by_rank[r].size() * sizeof(FaceAdjReply);
                if (bytes > 0) {
                    std::memcpy(send_buffer.data() + off, replies_by_rank[r].data(), bytes);
                    off += bytes;
                }
            }
        }

        recv_buffer.clear();
        alltoallv_records(send_buffer, send_counts, recv_buffer, recv_counts, recv_displs);

        // Build next layer requests from replies.
        std::unordered_map<rank_t, std::unordered_set<gid_t>> next_requests;
        for (int r = 0; r < world_size_; ++r) {
            const int n_bytes = recv_counts[r];
            const int n_recs = n_bytes / static_cast<int>(sizeof(FaceAdjReply));
            const auto* reps = reinterpret_cast<const FaceAdjReply*>(recv_buffer.data() + recv_displs[r]);

            for (int i = 0; i < n_recs; ++i) {
                const auto& rep = reps[i];
                if (rep.neighbor_cell_gid == INVALID_GID) continue;
                if (rep.neighbor_owner_rank < 0 || rep.neighbor_owner_rank == my_rank_) continue;
                if (local_mesh_->global_to_local_cell(rep.neighbor_cell_gid) != INVALID_INDEX) continue;
                next_requests[rep.neighbor_owner_rank].insert(rep.neighbor_cell_gid);
            }
        }

	        frontier = exchange_cells(next_requests);
	    }

	    // Ghost-layer rebuild invalidates any existing exchange patterns; rebuild them now so
	    // ghost/shared exchanges (fields/coordinates) use the correct post-ghost topology.
	    build_exchange_patterns();
	#endif
	}

#if defined(SVMP_HAS_PARMETIS)
	void DistributedMesh::build_from_arrays_global_and_partition_two_phase_parmetis_(
	    int spatial_dim,
	    const std::vector<real_t>& X_ref,
	    const std::vector<offset_t>& cell2vertex_offsets,
	    const std::vector<index_t>& cell2vertex,
	    const std::vector<CellShape>& cell_shape,
	    PartitionHint hint,
	    int ghost_layers,
	    const std::unordered_map<std::string, std::string>& options) {
#ifdef MESH_HAS_MPI
	    if (!local_mesh_) {
	        local_mesh_ = std::make_shared<MeshBase>();
	    }
	    if (world_size_ <= 1) {
	        build_from_arrays_local(spatial_dim, X_ref, cell2vertex_offsets, cell2vertex, cell_shape);
	        return;
	    }

	    auto checked_int_cast = [](std::uint64_t value, const char* what) -> int {
	        const auto max_int = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
	        if (value > max_int) {
	            throw std::runtime_error(std::string("DistributedMesh: ") + what + " exceeds MPI int count range");
	        }
	        return static_cast<int>(value);
	    };

	    auto block_start = [](std::uint64_t n, int p, int r) -> std::uint64_t {
	        if (p <= 0) return 0;
	        return (n * static_cast<std::uint64_t>(r)) / static_cast<std::uint64_t>(p);
	    };
	    auto block_end = [](std::uint64_t n, int p, int r) -> std::uint64_t {
	        if (p <= 0) return 0;
	        return (n * static_cast<std::uint64_t>(r + 1)) / static_cast<std::uint64_t>(p);
	    };

	    auto owner_rank_for_vertex_block = [&](index_t v, std::uint64_t global_n_vertices) -> rank_t {
	        if (v < 0) return 0;
	        const auto gv = static_cast<std::uint64_t>(v);
	        if (gv >= global_n_vertices || global_n_vertices == 0) return 0;
	        // Map v -> owning rank under the same [start,end) block partitioning used by block_start/end.
	        const auto r = ((gv + 1u) * static_cast<std::uint64_t>(world_size_) - 1u) / global_n_vertices;
	        return static_cast<rank_t>(std::min<std::uint64_t>(static_cast<std::uint64_t>(world_size_ - 1), r));
	    };

	    // -------------------------
	    // Step 0: broadcast sizes
	    // -------------------------
	    std::uint64_t global_n_cells = 0;
	    std::uint64_t global_n_vertices = 0;
	    if (my_rank_ == 0) {
	        if (spatial_dim < 1 || spatial_dim > 3) {
	            throw std::invalid_argument("DistributedMesh::two_phase_parmetis: invalid spatial_dim");
	        }
	        if (cell2vertex_offsets.empty() || cell2vertex_offsets[0] != 0) {
	            throw std::invalid_argument("DistributedMesh::two_phase_parmetis: invalid cell2vertex_offsets");
	        }
	        if (cell_shape.size() + 1 != cell2vertex_offsets.size()) {
	            throw std::invalid_argument("DistributedMesh::two_phase_parmetis: shape/offset size mismatch");
	        }
	        global_n_cells = static_cast<std::uint64_t>(cell_shape.size());

	        const std::uint64_t expected_vertices =
	            static_cast<std::uint64_t>(X_ref.size()) / static_cast<std::uint64_t>(std::max(1, spatial_dim));
	        if (static_cast<std::uint64_t>(X_ref.size()) != expected_vertices * static_cast<std::uint64_t>(spatial_dim)) {
	            throw std::invalid_argument("DistributedMesh::two_phase_parmetis: coordinate array size mismatch");
	        }
	        global_n_vertices = expected_vertices;

	        const auto expected_conn =
	            static_cast<std::uint64_t>(cell2vertex_offsets.back());
	        if (expected_conn != static_cast<std::uint64_t>(cell2vertex.size())) {
	            throw std::invalid_argument("DistributedMesh::two_phase_parmetis: connectivity size mismatch");
	        }
	    }

	    MPI_Bcast(&global_n_cells, 1, MPI_UINT64_T, 0, comm_);
	    MPI_Bcast(&global_n_vertices, 1, MPI_UINT64_T, 0, comm_);

	    if (global_n_cells == 0) {
	        local_mesh_->clear();
	        reset_partition_state_();
	        return;
	    }

	    // -------------------------
	    // Step 1: scatter connectivity (cells) by block (no coordinates)
	    // -------------------------
	    const std::uint64_t my_cell_start = block_start(global_n_cells, world_size_, my_rank_);
	    const std::uint64_t my_cell_end = block_end(global_n_cells, world_size_, my_rank_);
	    const std::uint64_t my_n_cells_u64 = (my_cell_end >= my_cell_start) ? (my_cell_end - my_cell_start) : 0;

	    std::vector<int> send_counts_offsets(world_size_, 0);
	    std::vector<int> send_displs_offsets(world_size_ + 1, 0);
	    std::vector<int> send_counts_shapes_bytes(world_size_, 0);
	    std::vector<int> send_displs_shapes_bytes(world_size_ + 1, 0);

	    for (int r = 0; r < world_size_; ++r) {
	        const std::uint64_t start = block_start(global_n_cells, world_size_, r);
	        const std::uint64_t end = block_end(global_n_cells, world_size_, r);
	        const std::uint64_t n_local = (end >= start) ? (end - start) : 0;

	        send_counts_offsets[r] = checked_int_cast(n_local + 1u, "cell2vertex_offsets scatter count");
	        send_displs_offsets[r] = checked_int_cast(start, "cell2vertex_offsets scatter displacement");

	        const std::uint64_t bytes = n_local * static_cast<std::uint64_t>(sizeof(CellShape));
	        const std::uint64_t disp_bytes = start * static_cast<std::uint64_t>(sizeof(CellShape));
	        send_counts_shapes_bytes[r] = checked_int_cast(bytes, "cell_shape scatter byte count");
	        send_displs_shapes_bytes[r] = checked_int_cast(disp_bytes, "cell_shape scatter byte displacement");
	    }
	    send_displs_offsets[world_size_] = 0;
	    send_displs_shapes_bytes[world_size_] = 0;

	    std::vector<int> send_counts_conn(world_size_, 0);
	    std::vector<int> send_displs_conn(world_size_ + 1, 0);
	    if (my_rank_ == 0) {
	        for (int r = 0; r < world_size_; ++r) {
	            const std::uint64_t start = block_start(global_n_cells, world_size_, r);
	            const std::uint64_t end = block_end(global_n_cells, world_size_, r);
	            const std::uint64_t conn_start = static_cast<std::uint64_t>(cell2vertex_offsets[static_cast<size_t>(start)]);
	            const std::uint64_t conn_end = static_cast<std::uint64_t>(cell2vertex_offsets[static_cast<size_t>(end)]);
	            if (conn_end < conn_start) {
	                throw std::runtime_error("DistributedMesh::two_phase_parmetis: invalid connectivity offsets");
	            }
	            send_counts_conn[r] = checked_int_cast(conn_end - conn_start, "cell2vertex scatter count");
	            send_displs_conn[r] = checked_int_cast(conn_start, "cell2vertex scatter displacement");
	        }
	    }

	    MPI_Bcast(send_counts_conn.data(), world_size_, MPI_INT, 0, comm_);
	    MPI_Bcast(send_displs_conn.data(), world_size_, MPI_INT, 0, comm_);

	    const int my_conn_count = send_counts_conn[static_cast<size_t>(my_rank_)];

	    std::vector<CellShape> local_shapes(static_cast<size_t>(my_n_cells_u64));
	    std::vector<offset_t> local_offsets_global(static_cast<size_t>(my_n_cells_u64) + 1u, 0);
	    std::vector<index_t> local_conn(static_cast<size_t>(std::max(0, my_conn_count)));

	    static_assert(std::is_trivially_copyable<CellShape>::value, "CellShape must be trivially copyable for MPI_BYTE");

	    MPI_Scatterv(my_rank_ == 0 ? reinterpret_cast<const void*>(cell_shape.data()) : nullptr,
	                 send_counts_shapes_bytes.data(),
	                 send_displs_shapes_bytes.data(),
	                 MPI_BYTE,
	                 reinterpret_cast<void*>(local_shapes.data()),
	                 checked_int_cast(my_n_cells_u64 * static_cast<std::uint64_t>(sizeof(CellShape)),
	                                  "local cell_shape recv bytes"),
	                 MPI_BYTE,
	                 0,
	                 comm_);

	    MPI_Scatterv(my_rank_ == 0 ? reinterpret_cast<const void*>(cell2vertex_offsets.data()) : nullptr,
	                 send_counts_offsets.data(),
	                 send_displs_offsets.data(),
	                 MPI_INT64_T,
	                 reinterpret_cast<void*>(local_offsets_global.data()),
	                 checked_int_cast(my_n_cells_u64 + 1u, "local cell2vertex_offsets recv count"),
	                 MPI_INT64_T,
	                 0,
	                 comm_);

	    MPI_Scatterv(my_rank_ == 0 ? reinterpret_cast<const void*>(cell2vertex.data()) : nullptr,
	                 send_counts_conn.data(),
	                 send_displs_conn.data(),
	                 MPI_INT32_T,
	                 reinterpret_cast<void*>(local_conn.data()),
	                 checked_int_cast(static_cast<std::uint64_t>(std::max(0, my_conn_count)),
	                                  "local cell2vertex recv count"),
	                 MPI_INT32_T,
	                 0,
	                 comm_);

	    std::vector<offset_t> local_offsets(local_offsets_global.size(), 0);
	    const offset_t base_offset = local_offsets_global.empty() ? 0 : local_offsets_global[0];
	    for (size_t i = 0; i < local_offsets_global.size(); ++i) {
	        local_offsets[i] = local_offsets_global[i] - base_offset;
	    }
	    if (!local_offsets.empty() && local_offsets.front() != 0) {
	        throw std::runtime_error("DistributedMesh::two_phase_parmetis: local offsets do not start at 0");
	    }
	    const std::uint64_t expected_local_conn =
	        local_offsets.empty() ? 0 : static_cast<std::uint64_t>(local_offsets.back());
	    if (expected_local_conn != static_cast<std::uint64_t>(local_conn.size())) {
	        throw std::runtime_error("DistributedMesh::two_phase_parmetis: local connectivity size mismatch");
	    }

	    std::vector<gid_t> local_cell_gids(static_cast<size_t>(my_n_cells_u64), INVALID_GID);
	    for (std::uint64_t i = 0; i < my_n_cells_u64; ++i) {
	        local_cell_gids[static_cast<size_t>(i)] = static_cast<gid_t>(my_cell_start + i);
	    }

	    // -------------------------
	    // Step 2: ParMETIS partition (mesh algorithm) on distributed connectivity
	    // -------------------------
	    const std::uint64_t my_local_cells = my_n_cells_u64;
	    std::vector<std::uint64_t> cell_counts(static_cast<size_t>(world_size_), 0);
	    MPI_Allgather(&my_local_cells, 1, MPI_UINT64_T,
	                  cell_counts.data(), 1, MPI_UINT64_T, comm_);

	    std::vector<int> active_ranks;
	    active_ranks.reserve(static_cast<size_t>(world_size_));
	    for (int r = 0; r < world_size_; ++r) {
	        if (cell_counts[static_cast<size_t>(r)] > 0) {
	            active_ranks.push_back(r);
	        }
	    }
	    const int active_size = static_cast<int>(active_ranks.size());
	    const bool is_active = (my_n_cells_u64 > 0);

	    MPI_Comm pm_comm = comm_;
	    MPI_Comm active_comm = MPI_COMM_NULL;
	    std::vector<int> active_index(static_cast<size_t>(world_size_), -1);
	    for (int i = 0; i < active_size; ++i) {
	        active_index[static_cast<size_t>(active_ranks[static_cast<size_t>(i)])] = i;
	    }

	    if (active_size >= 2 && active_size < world_size_) {
	        MPI_Group world_group;
	        MPI_Comm_group(comm_, &world_group);
	        MPI_Group incl_group;
	        MPI_Group_incl(world_group, active_size, active_ranks.data(), &incl_group);
	        MPI_Comm_create(comm_, incl_group, &active_comm);
	        MPI_Group_free(&incl_group);
	        MPI_Group_free(&world_group);
	        if (active_comm != MPI_COMM_NULL) {
	            pm_comm = active_comm;
	        }
	    }

	    std::vector<rank_t> new_owner_rank_per_cell(static_cast<size_t>(my_n_cells_u64), my_rank_);

	    int mesh_status = METIS_ERROR;
	    if (active_size >= 2 && is_active && pm_comm != MPI_COMM_NULL) {
	        std::vector<::idx_t> elmdist(static_cast<size_t>(active_size) + 1u, 0);
	        std::uint64_t prefix = 0;
	        elmdist[0] = 0;
	        for (int i = 0; i < active_size; ++i) {
	            prefix += cell_counts[static_cast<size_t>(active_ranks[static_cast<size_t>(i)])];
	            elmdist[static_cast<size_t>(i) + 1u] = checked_idx_cast(prefix, "ParMETIS elmdist");
	        }

	        const size_t n_owned = static_cast<size_t>(my_n_cells_u64);
	        std::vector<::idx_t> eptr(n_owned + 1u, 0);
	        for (size_t i = 0; i < eptr.size(); ++i) {
	            const auto off = (i < local_offsets.size()) ? static_cast<std::uint64_t>(local_offsets[i]) : 0;
	            eptr[i] = checked_idx_cast(off, "ParMETIS eptr");
	        }

	        std::vector<::idx_t> eind(local_conn.size(), 0);
	        for (size_t i = 0; i < local_conn.size(); ++i) {
	            const auto v = local_conn[i];
	            if (v < 0) {
	                throw std::runtime_error("DistributedMesh::two_phase_parmetis: negative vertex index in connectivity");
	            }
	            eind[i] = checked_idx_cast(static_cast<std::uint64_t>(v), "ParMETIS eind");
	        }

	        ::idx_t ncommonnodes = 0;
	        if (const auto it = options.find("parmetis_ncommonnodes"); it != options.end()) {
	            ncommonnodes = static_cast<::idx_t>(std::stoi(it->second));
	        } else {
	            int local_min = std::numeric_limits<int>::max();
	            for (size_t c = 0; c < n_owned; ++c) {
	                const auto& cs = local_shapes[c];
	                const size_t nv = static_cast<size_t>(local_offsets[c + 1u] - local_offsets[c]);

	                int p = std::max(1, cs.order);
	                CellTopology::HighOrderKind kind = CellTopology::HighOrderKind::Lagrange;
	                if (nv > static_cast<size_t>(std::max(0, cs.num_corners))) {
	                    const int p_ser = CellTopology::infer_serendipity_order(cs.family, nv);
	                    const int p_lag = CellTopology::infer_lagrange_order(cs.family, nv);
	                    if (p_ser > 0) {
	                        p = p_ser;
	                        kind = CellTopology::HighOrderKind::Serendipity;
	                    } else if (p_lag > 0) {
	                        p = p_lag;
	                    }
	                }

	                int cell_min = std::numeric_limits<int>::max();
	                try {
	                    const auto faces = CellTopology::get_oriented_boundary_faces(cs.family);
	                    for (int f = 0; f < static_cast<int>(faces.size()); ++f) {
	                        int n = 0;
	                        if (p <= 1) {
	                            n = static_cast<int>(faces[static_cast<size_t>(f)].size());
	                        } else {
	                            n = static_cast<int>(CellTopology::high_order_face_local_nodes(cs.family, p, f, kind).size());
	                        }
	                        cell_min = std::min(cell_min, n);
	                    }
	                } catch (...) {
	                    cell_min = std::max(1, spatial_dim);
	                }
	                if (cell_min == std::numeric_limits<int>::max()) {
	                    cell_min = std::max(1, spatial_dim);
	                }
	                local_min = std::min(local_min, cell_min);
	            }

	            if (local_min == std::numeric_limits<int>::max()) {
	                local_min = std::max(1, spatial_dim);
	            }

	            int global_min = local_min;
	            MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, pm_comm);
	            ncommonnodes = checked_idx_cast(static_cast<std::uint64_t>(std::max(1, global_min)),
	                                            "ParMETIS ncommonnodes");
	        }

	        PartitionHint weight_hint = hint;
	        if (weight_hint == PartitionHint::Metis || weight_hint == PartitionHint::ParMetis) {
	            weight_hint = PartitionHint::Cells;
	        }
	        if (const auto it = options.find("partition_weight"); it != options.end()) {
	            std::string w = it->second;
	            std::transform(w.begin(), w.end(), w.begin(),
	                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	            if (w == "cells" || w == "cell") {
	                weight_hint = PartitionHint::Cells;
	            } else if (w == "vertices" || w == "vertex") {
	                weight_hint = PartitionHint::Vertices;
	            } else if (w == "memory" || w == "mem") {
	                weight_hint = PartitionHint::Memory;
	            }
	        }

	        std::vector<::idx_t> elmwgt;
	        ::idx_t wgtflag = 0;
	        ::idx_t* elmwgt_ptr = nullptr;
	        if (weight_hint == PartitionHint::Vertices || weight_hint == PartitionHint::Memory) {
	            wgtflag = 2;
	            elmwgt.resize(n_owned, 1);
	            for (size_t i = 0; i < n_owned; ++i) {
	                const std::uint64_t nv = static_cast<std::uint64_t>(local_offsets[i + 1u] - local_offsets[i]);
	                elmwgt[i] = checked_idx_cast(std::max<std::uint64_t>(1, nv), "ParMETIS elmwgt");
	            }
	            elmwgt_ptr = elmwgt.data();
	        }

	        ::idx_t numflag = 0;
	        ::idx_t ncon = 1;
	        ::idx_t nparts = checked_idx_cast(static_cast<std::uint64_t>(active_size), "ParMETIS nparts");

	        std::vector<::real_t> tpwgts(static_cast<size_t>(ncon) * static_cast<size_t>(nparts),
	                                     static_cast<::real_t>(1.0) / static_cast<::real_t>(nparts));
	        std::vector<::real_t> ubvec(static_cast<size_t>(ncon), static_cast<::real_t>(1.05));

	        std::array<::idx_t, 4> pm_options{{0, 0, 0, 0}};
	        pm_options[0] = 1;
	        pm_options[1] = 0;
	        pm_options[2] = 42;
	        pm_options[3] = 0;

	        std::vector<::idx_t> part(n_owned, checked_idx_cast(static_cast<std::uint64_t>(active_index[static_cast<size_t>(my_rank_)]),
	                                                           "ParMETIS initial part"));
	        ::idx_t edgecut = 0;
	        MPI_Comm comm = pm_comm;

	        mesh_status =
	            ParMETIS_V3_PartMeshKway(elmdist.data(),
	                                     eptr.data(),
	                                     eind.data(),
	                                     elmwgt_ptr,
	                                     &wgtflag,
	                                     &numflag,
	                                     &ncon,
	                                     &ncommonnodes,
	                                     &nparts,
	                                     tpwgts.data(),
	                                     ubvec.data(),
	                                     pm_options.data(),
	                                     &edgecut,
	                                     part.data(),
	                                     &comm);

	        if (mesh_status == METIS_OK) {
	            for (size_t i = 0; i < n_owned; ++i) {
	                const auto p = part[i];
	                if (p < 0 || p >= active_size) continue;
	                new_owner_rank_per_cell[i] = static_cast<rank_t>(active_ranks[static_cast<size_t>(p)]);
	            }
	        }
	    }

	    int local_ok = 1;
	    if (active_size >= 2 && is_active) {
	        local_ok = (mesh_status == METIS_OK) ? 1 : 0;
	    }
	    int global_ok = 0;
	    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm_);

	    if (active_comm != MPI_COMM_NULL) {
	        MPI_Comm_free(&active_comm);
	    }

	    if (active_size >= 2 && global_ok == 0) {
	        // Safe fallback: keep initial block distribution.
	        std::fill(new_owner_rank_per_cell.begin(), new_owner_rank_per_cell.end(), my_rank_);
	    }

	    // -------------------------
	    // Step 3: migrate cells (shape + connectivity + cell GIDs) based on partition
	    // -------------------------
	    std::vector<int> send_cell_counts(world_size_, 0);
	    std::vector<std::uint64_t> send_conn_counts_u64(static_cast<size_t>(world_size_), 0);
	    for (std::uint64_t c = 0; c < my_n_cells_u64; ++c) {
	        const auto dest = new_owner_rank_per_cell[static_cast<size_t>(c)];
	        const int r = (dest >= 0 && dest < world_size_) ? dest : my_rank_;
	        send_cell_counts[r] += 1;
	        const std::uint64_t nv = static_cast<std::uint64_t>(local_offsets[static_cast<size_t>(c) + 1u] -
	                                                            local_offsets[static_cast<size_t>(c)]);
	        send_conn_counts_u64[static_cast<size_t>(r)] += nv;
	    }

	    std::vector<int> send_conn_counts(world_size_, 0);
	    for (int r = 0; r < world_size_; ++r) {
	        send_conn_counts[r] = checked_int_cast(send_conn_counts_u64[static_cast<size_t>(r)], "cell2vertex migrate count");
	    }

	    std::vector<int> recv_cell_counts(world_size_, 0);
	    std::vector<int> recv_conn_counts(world_size_, 0);
	    MPI_Alltoall(send_cell_counts.data(), 1, MPI_INT, recv_cell_counts.data(), 1, MPI_INT, comm_);
	    MPI_Alltoall(send_conn_counts.data(), 1, MPI_INT, recv_conn_counts.data(), 1, MPI_INT, comm_);

	    std::vector<int> send_cell_displs(world_size_ + 1, 0);
	    std::vector<int> recv_cell_displs(world_size_ + 1, 0);
	    std::vector<int> send_conn_displs(world_size_ + 1, 0);
	    std::vector<int> recv_conn_displs(world_size_ + 1, 0);

	    for (int r = 0; r < world_size_; ++r) {
	        send_cell_displs[r + 1] = send_cell_displs[r] + send_cell_counts[r];
	        recv_cell_displs[r + 1] = recv_cell_displs[r] + recv_cell_counts[r];
	        send_conn_displs[r + 1] = send_conn_displs[r] + send_conn_counts[r];
	        recv_conn_displs[r + 1] = recv_conn_displs[r] + recv_conn_counts[r];
	    }

	    const int total_send_cells = send_cell_displs[world_size_];
	    const int total_recv_cells = recv_cell_displs[world_size_];
	    const int total_send_conn = send_conn_displs[world_size_];
	    const int total_recv_conn = recv_conn_displs[world_size_];

	    std::vector<gid_t> send_cell_gids(static_cast<size_t>(total_send_cells), INVALID_GID);
	    std::vector<CellShape> send_cell_shapes(static_cast<size_t>(total_send_cells));
	    std::vector<std::int32_t> send_cell_nv(static_cast<size_t>(total_send_cells), 0);
	    std::vector<index_t> send_cell_conn(static_cast<size_t>(total_send_conn), 0);

	    std::vector<int> cell_cursor = send_cell_displs;
	    std::vector<int> conn_cursor = send_conn_displs;
	    for (std::uint64_t c = 0; c < my_n_cells_u64; ++c) {
	        const auto dest = new_owner_rank_per_cell[static_cast<size_t>(c)];
	        const int r = (dest >= 0 && dest < world_size_) ? dest : my_rank_;

	        const int cell_pos = cell_cursor[r]++;
	        send_cell_gids[static_cast<size_t>(cell_pos)] = local_cell_gids[static_cast<size_t>(c)];
	        send_cell_shapes[static_cast<size_t>(cell_pos)] = local_shapes[static_cast<size_t>(c)];

	        const offset_t start = local_offsets[static_cast<size_t>(c)];
	        const offset_t end = local_offsets[static_cast<size_t>(c) + 1u];
	        const std::uint64_t nv_u64 = static_cast<std::uint64_t>(end - start);
	        send_cell_nv[static_cast<size_t>(cell_pos)] = static_cast<std::int32_t>(nv_u64);

	        const int conn_pos = conn_cursor[r];
	        const size_t src_off = static_cast<size_t>(start);
	        const size_t dst_off = static_cast<size_t>(conn_pos);
	        if (nv_u64 > 0) {
	            std::memcpy(send_cell_conn.data() + dst_off,
	                        local_conn.data() + src_off,
	                        static_cast<size_t>(nv_u64) * sizeof(index_t));
	        }
	        conn_cursor[r] += static_cast<int>(nv_u64);
	    }

	    std::vector<gid_t> recv_cell_gids(static_cast<size_t>(total_recv_cells), INVALID_GID);
	    std::vector<CellShape> recv_cell_shapes(static_cast<size_t>(total_recv_cells));
	    std::vector<std::int32_t> recv_cell_nv(static_cast<size_t>(total_recv_cells), 0);
	    std::vector<index_t> recv_cell_conn(static_cast<size_t>(total_recv_conn), 0);

	    MPI_Alltoallv(send_cell_gids.data(), send_cell_counts.data(), send_cell_displs.data(), MPI_INT64_T,
	                  recv_cell_gids.data(), recv_cell_counts.data(), recv_cell_displs.data(), MPI_INT64_T,
	                  comm_);

	    std::vector<int> send_shape_counts_bytes(world_size_, 0);
	    std::vector<int> recv_shape_counts_bytes(world_size_, 0);
	    std::vector<int> send_shape_displs_bytes(world_size_ + 1, 0);
	    std::vector<int> recv_shape_displs_bytes(world_size_ + 1, 0);
	    for (int r = 0; r < world_size_; ++r) {
	        send_shape_counts_bytes[r] = checked_int_cast(static_cast<std::uint64_t>(send_cell_counts[r]) *
	                                                         static_cast<std::uint64_t>(sizeof(CellShape)),
	                                                     "cell_shape migrate byte count");
	        recv_shape_counts_bytes[r] = checked_int_cast(static_cast<std::uint64_t>(recv_cell_counts[r]) *
	                                                         static_cast<std::uint64_t>(sizeof(CellShape)),
	                                                     "cell_shape migrate recv byte count");
	        send_shape_displs_bytes[r] = checked_int_cast(static_cast<std::uint64_t>(send_cell_displs[r]) *
	                                                         static_cast<std::uint64_t>(sizeof(CellShape)),
	                                                     "cell_shape migrate byte displacement");
	        recv_shape_displs_bytes[r] = checked_int_cast(static_cast<std::uint64_t>(recv_cell_displs[r]) *
	                                                         static_cast<std::uint64_t>(sizeof(CellShape)),
	                                                     "cell_shape migrate recv byte displacement");
	    }

	    MPI_Alltoallv(reinterpret_cast<const void*>(send_cell_shapes.data()),
	                  send_shape_counts_bytes.data(),
	                  send_shape_displs_bytes.data(),
	                  MPI_BYTE,
	                  reinterpret_cast<void*>(recv_cell_shapes.data()),
	                  recv_shape_counts_bytes.data(),
	                  recv_shape_displs_bytes.data(),
	                  MPI_BYTE,
	                  comm_);

	    MPI_Alltoallv(send_cell_nv.data(), send_cell_counts.data(), send_cell_displs.data(), MPI_INT32_T,
	                  recv_cell_nv.data(), recv_cell_counts.data(), recv_cell_displs.data(), MPI_INT32_T,
	                  comm_);

	    MPI_Alltoallv(send_cell_conn.data(), send_conn_counts.data(), send_conn_displs.data(), MPI_INT32_T,
	                  recv_cell_conn.data(), recv_conn_counts.data(), recv_conn_displs.data(), MPI_INT32_T,
	                  comm_);

	    std::vector<offset_t> final_offsets(static_cast<size_t>(total_recv_cells) + 1u, 0);
	    for (int i = 0; i < total_recv_cells; ++i) {
	        const std::int64_t nv = static_cast<std::int64_t>(recv_cell_nv[static_cast<size_t>(i)]);
	        final_offsets[static_cast<size_t>(i) + 1u] =
	            final_offsets[static_cast<size_t>(i)] + static_cast<offset_t>(std::max<std::int64_t>(0, nv));
	    }
	    if (!final_offsets.empty() &&
	        static_cast<std::uint64_t>(final_offsets.back()) != static_cast<std::uint64_t>(recv_cell_conn.size())) {
	        throw std::runtime_error("DistributedMesh::two_phase_parmetis: migrated connectivity size mismatch");
	    }

	    // -------------------------
	    // Step 4: vertex coordinates distribution by vertex-block ownership
	    // -------------------------
	    const std::uint64_t my_v_start = block_start(global_n_vertices, world_size_, my_rank_);
	    const std::uint64_t my_v_end = block_end(global_n_vertices, world_size_, my_rank_);
	    const std::uint64_t my_owned_vertices_u64 = (my_v_end >= my_v_start) ? (my_v_end - my_v_start) : 0;

	    std::vector<int> coord_send_counts(world_size_, 0);
	    std::vector<int> coord_send_displs(world_size_ + 1, 0);
	    for (int r = 0; r < world_size_; ++r) {
	        const std::uint64_t vs = block_start(global_n_vertices, world_size_, r);
	        const std::uint64_t ve = block_end(global_n_vertices, world_size_, r);
	        const std::uint64_t nverts = (ve >= vs) ? (ve - vs) : 0;
	        coord_send_counts[r] =
	            checked_int_cast(nverts * static_cast<std::uint64_t>(spatial_dim), "coordinate scatter count");
	        coord_send_displs[r] =
	            checked_int_cast(vs * static_cast<std::uint64_t>(spatial_dim), "coordinate scatter displacement");
	    }

	    std::vector<real_t> owned_coords(my_owned_vertices_u64 * static_cast<std::uint64_t>(spatial_dim), 0.0);
	    MPI_Scatterv(my_rank_ == 0 ? reinterpret_cast<const void*>(X_ref.data()) : nullptr,
	                 coord_send_counts.data(),
	                 coord_send_displs.data(),
	                 MPI_DOUBLE,
	                 reinterpret_cast<void*>(owned_coords.data()),
	                 checked_int_cast(my_owned_vertices_u64 * static_cast<std::uint64_t>(spatial_dim),
	                                  "owned coordinate recv count"),
	                 MPI_DOUBLE,
	                 0,
	                 comm_);

	    // Local vertex set needed by owned cells.
	    std::vector<index_t> needed_vertices;
	    needed_vertices.reserve(recv_cell_conn.size());
	    for (const auto v : recv_cell_conn) {
	        needed_vertices.push_back(v);
	    }
	    std::sort(needed_vertices.begin(), needed_vertices.end());
	    needed_vertices.erase(std::unique(needed_vertices.begin(), needed_vertices.end()), needed_vertices.end());

	    std::vector<std::vector<index_t>> req_by_rank(world_size_);
	    for (const auto v : needed_vertices) {
	        const auto owner = owner_rank_for_vertex_block(v, global_n_vertices);
	        if (owner == my_rank_) continue;
	        if (owner < 0 || owner >= world_size_) continue;
	        req_by_rank[static_cast<size_t>(owner)].push_back(v);
	    }

	    std::vector<int> send_req_counts(world_size_, 0);
	    std::vector<int> send_req_displs(world_size_ + 1, 0);
	    for (int r = 0; r < world_size_; ++r) {
	        send_req_counts[r] = checked_int_cast(static_cast<std::uint64_t>(req_by_rank[static_cast<size_t>(r)].size()),
	                                              "vertex coordinate request count");
	        send_req_displs[r + 1] = send_req_displs[r] + send_req_counts[r];
	    }

	    std::vector<index_t> send_req_vertices(static_cast<size_t>(send_req_displs[world_size_]), 0);
	    for (int r = 0; r < world_size_; ++r) {
	        int pos = send_req_displs[r];
	        for (const auto v : req_by_rank[static_cast<size_t>(r)]) {
	            send_req_vertices[static_cast<size_t>(pos++)] = v;
	        }
	    }

	    std::vector<int> recv_req_counts(world_size_, 0);
	    std::vector<int> recv_req_displs(world_size_ + 1, 0);
	    MPI_Alltoall(send_req_counts.data(), 1, MPI_INT, recv_req_counts.data(), 1, MPI_INT, comm_);
	    for (int r = 0; r < world_size_; ++r) {
	        recv_req_displs[r + 1] = recv_req_displs[r] + recv_req_counts[r];
	    }

	    std::vector<index_t> recv_req_vertices(static_cast<size_t>(recv_req_displs[world_size_]), 0);
	    MPI_Alltoallv(send_req_vertices.data(), send_req_counts.data(), send_req_displs.data(), MPI_INT32_T,
	                  recv_req_vertices.data(), recv_req_counts.data(), recv_req_displs.data(), MPI_INT32_T,
	                  comm_);

	    std::vector<int> send_rep_counts(world_size_, 0);
	    std::vector<int> send_rep_displs(world_size_ + 1, 0);
	    for (int r = 0; r < world_size_; ++r) {
	        send_rep_counts[r] = checked_int_cast(static_cast<std::uint64_t>(recv_req_counts[r]) *
	                                                  static_cast<std::uint64_t>(spatial_dim),
	                                              "vertex coordinate reply count");
	        send_rep_displs[r + 1] = send_rep_displs[r] + send_rep_counts[r];
	    }

	    std::vector<real_t> send_rep_coords(static_cast<size_t>(send_rep_displs[world_size_]), 0.0);
	    for (int r = 0; r < world_size_; ++r) {
	        const int n = recv_req_counts[r];
	        const int in_off = recv_req_displs[r];
	        const int out_off = send_rep_displs[r];
	        for (int i = 0; i < n; ++i) {
	            const index_t v = recv_req_vertices[static_cast<size_t>(in_off + i)];
	            const std::uint64_t gv = static_cast<std::uint64_t>(v);
	            if (gv < my_v_start || gv >= my_v_end) {
	                continue;
	            }
	            const std::uint64_t local = gv - my_v_start;
	            const size_t src = static_cast<size_t>(local * static_cast<std::uint64_t>(spatial_dim));
	            const size_t dst = static_cast<size_t>((out_off + i * spatial_dim));
	            for (int d = 0; d < spatial_dim; ++d) {
	                send_rep_coords[dst + static_cast<size_t>(d)] = owned_coords[src + static_cast<size_t>(d)];
	            }
	        }
	    }

	    std::vector<int> recv_rep_counts(world_size_, 0);
	    std::vector<int> recv_rep_displs(world_size_ + 1, 0);
	    for (int r = 0; r < world_size_; ++r) {
	        recv_rep_counts[r] = checked_int_cast(static_cast<std::uint64_t>(send_req_counts[r]) *
	                                                  static_cast<std::uint64_t>(spatial_dim),
	                                              "vertex coordinate receive count");
	        recv_rep_displs[r + 1] = recv_rep_displs[r] + recv_rep_counts[r];
	    }

	    std::vector<real_t> recv_rep_coords(static_cast<size_t>(recv_rep_displs[world_size_]), 0.0);
	    MPI_Alltoallv(send_rep_coords.data(), send_rep_counts.data(), send_rep_displs.data(), MPI_DOUBLE,
	                  recv_rep_coords.data(), recv_rep_counts.data(), recv_rep_displs.data(), MPI_DOUBLE,
	                  comm_);

	    std::unordered_map<index_t, size_t> remote_coord_offset;
	    remote_coord_offset.reserve(static_cast<size_t>(send_req_displs[world_size_]) + 16u);
	    for (int r = 0; r < world_size_; ++r) {
	        if (r == my_rank_) continue;
	        const auto& req = req_by_rank[static_cast<size_t>(r)];
	        const int base = recv_rep_displs[r];
	        for (size_t i = 0; i < req.size(); ++i) {
	            remote_coord_offset[req[i]] =
	                static_cast<size_t>(base + static_cast<int>(i) * spatial_dim);
	        }
	    }

	    std::vector<real_t> local_X_ref(static_cast<size_t>(needed_vertices.size()) *
	                                    static_cast<size_t>(spatial_dim), 0.0);
	    for (size_t i = 0; i < needed_vertices.size(); ++i) {
	        const index_t v = needed_vertices[i];
	        const auto owner = owner_rank_for_vertex_block(v, global_n_vertices);
	        const size_t dst = i * static_cast<size_t>(spatial_dim);
	        if (owner == my_rank_) {
	            const std::uint64_t gv = static_cast<std::uint64_t>(v);
	            if (gv < my_v_start || gv >= my_v_end) {
	                continue;
	            }
	            const std::uint64_t local = gv - my_v_start;
	            const size_t src = static_cast<size_t>(local * static_cast<std::uint64_t>(spatial_dim));
	            for (int d = 0; d < spatial_dim; ++d) {
	                local_X_ref[dst + static_cast<size_t>(d)] = owned_coords[src + static_cast<size_t>(d)];
	            }
	        } else {
	            const auto it = remote_coord_offset.find(v);
	            if (it == remote_coord_offset.end()) {
	                throw std::runtime_error("DistributedMesh::two_phase_parmetis: missing remote vertex coordinates");
	            }
	            const size_t src = it->second;
	            for (int d = 0; d < spatial_dim; ++d) {
	                local_X_ref[dst + static_cast<size_t>(d)] = recv_rep_coords[src + static_cast<size_t>(d)];
	            }
	        }
	    }

	    // -------------------------
	    // Step 5: build local MeshBase (post-partition) on each rank
	    // -------------------------
	    std::vector<gid_t> local_vertex_gids;
	    local_vertex_gids.reserve(needed_vertices.size());
	    std::unordered_map<index_t, index_t> vertex_gid_to_local;
	    vertex_gid_to_local.reserve(static_cast<size_t>(needed_vertices.size()) * 2u + 1u);
	    for (index_t i = 0; i < static_cast<index_t>(needed_vertices.size()); ++i) {
	        const index_t gid = needed_vertices[static_cast<size_t>(i)];
	        local_vertex_gids.push_back(static_cast<gid_t>(gid));
	        vertex_gid_to_local.emplace(gid, i);
	    }

	    std::vector<index_t> local_cell2vertex = recv_cell_conn;
	    for (size_t i = 0; i < local_cell2vertex.size(); ++i) {
	        const index_t gv = local_cell2vertex[i];
	        const auto it = vertex_gid_to_local.find(gv);
	        if (it == vertex_gid_to_local.end()) {
	            throw std::runtime_error("DistributedMesh::two_phase_parmetis: missing vertex in local map");
	        }
	        local_cell2vertex[i] = it->second;
	    }

	    local_mesh_->clear();
	    local_mesh_->build_from_arrays(spatial_dim, local_X_ref, final_offsets, local_cell2vertex, recv_cell_shapes);
	    local_mesh_->set_vertex_gids(std::move(local_vertex_gids));
	    local_mesh_->set_cell_gids(std::move(recv_cell_gids));
	    local_mesh_->finalize();

	    // -------------------------
	    // Step 6: build ownership/exchange + optional ghost layers
	    // -------------------------
	    reset_partition_state_();
	    if (ghost_layers > 0) {
	        build_ghost_layer(ghost_layers);
	    } else {
	        build_exchange_patterns();
	    }
#else
	    (void)spatial_dim;
	    (void)X_ref;
	    (void)cell2vertex_offsets;
	    (void)cell2vertex;
	    (void)cell_shape;
	    (void)hint;
	    (void)ghost_layers;
	    (void)options;
#endif
	}
#endif

void DistributedMesh::clear_ghosts() {
    ghost_vertices_.clear();
    ghost_cells_.clear();
    ghost_faces_.clear();
    ghost_edges_.clear();
    ghost_levels_ = 0;

#ifdef MESH_HAS_MPI
    auto notify_partition_changed = [&]() {
        if (local_mesh_) {
            local_mesh_->event_bus().notify(MeshEvent::PartitionChanged);
        }
    };

    if (world_size_ == 1) {
        notify_partition_changed();
        return;
    }

    // Filter out ghost cells (keep locally owned cells only).
    std::vector<index_t> keep_cells;
    keep_cells.reserve(local_mesh_->n_cells());
    for (index_t c = 0; c < static_cast<index_t>(local_mesh_->n_cells()); ++c) {
        if (c < static_cast<index_t>(cell_owner_rank_.size()) && cell_owner_rank_[c] == my_rank_) {
            keep_cells.push_back(c);
        }
    }

    const bool local_has_ghosts = (keep_cells.size() != local_mesh_->n_cells());

    // clear_ghosts() is a collective operation. If any rank has ghosts, all ranks
    // must participate in the shared-entity detection collectives to avoid hangs.
    int local_has = local_has_ghosts ? 1 : 0;
    int any_has = 0;
    MPI_Allreduce(&local_has, &any_has, 1, MPI_INT, MPI_MAX, comm_);

    if (any_has == 0) {
        // No ghosts present anywhere; preserve existing shared/owned state.
        notify_partition_changed();
        return;
    }

    if (!local_has_ghosts) {
        // No local rebuild needed, but still re-establish base ownership semantics so
        // all ranks remain consistent after other ranks drop ghosts.
        vertex_owner_.assign(local_mesh_->n_vertices(), Ownership::Owned);
        vertex_owner_rank_.assign(local_mesh_->n_vertices(), my_rank_);
        cell_owner_.assign(local_mesh_->n_cells(), Ownership::Owned);
        cell_owner_rank_.assign(local_mesh_->n_cells(), my_rank_);
        face_owner_.assign(local_mesh_->n_faces(), Ownership::Owned);
        face_owner_rank_.assign(local_mesh_->n_faces(), my_rank_);
        edge_owner_.assign(local_mesh_->n_edges(), Ownership::Owned);
        edge_owner_rank_.assign(local_mesh_->n_edges(), my_rank_);

        gather_shared_entities();
        notify_partition_changed();
        return;
    }

	    // Collect vertices used by kept cells.
	    std::unordered_set<index_t> keep_vertex_set;
	    for (index_t c : keep_cells) {
	        auto [verts, n] = local_mesh_->cell_vertices_span(c);
	        for (size_t i = 0; i < n; ++i) keep_vertex_set.insert(verts[i]);
	    }

	    std::vector<index_t> keep_vertices(keep_vertex_set.begin(), keep_vertex_set.end());
	    std::sort(keep_vertices.begin(), keep_vertices.end());

	    std::unordered_map<index_t, index_t> old2new_vertex;
	    old2new_vertex.reserve(keep_vertices.size());
	    for (size_t i = 0; i < keep_vertices.size(); ++i) {
	        old2new_vertex[keep_vertices[i]] = static_cast<index_t>(i);
	    }

	    std::unordered_map<index_t, index_t> old2new_cell;
	    old2new_cell.reserve(keep_cells.size());
	    for (size_t i = 0; i < keep_cells.size(); ++i) {
	        old2new_cell[keep_cells[i]] = static_cast<index_t>(i);
	    }

	    const auto label_registry = local_mesh_->list_label_names();
	    const Configuration old_active_config = local_mesh_->active_configuration();
	    const bool old_has_current_coords = local_mesh_->has_current_coords();

	    const int dim = local_mesh_->dim();
	    std::vector<real_t> new_coords;
	    new_coords.reserve(keep_vertices.size() * static_cast<size_t>(dim));
	    std::vector<real_t> new_current_coords;
	    if (old_has_current_coords) {
	        new_current_coords.reserve(keep_vertices.size() * static_cast<size_t>(dim));
	    }
	    std::vector<std::pair<index_t, label_t>> kept_vertex_labels;
	    kept_vertex_labels.reserve(keep_vertices.size());
	    for (index_t v : keep_vertices) {
	        for (int d = 0; d < dim; ++d) {
	            new_coords.push_back(local_mesh_->X_ref()[static_cast<size_t>(v) * dim + d]);
	        }
	        if (old_has_current_coords) {
	            for (int d = 0; d < dim; ++d) {
	                new_current_coords.push_back(local_mesh_->X_cur()[static_cast<size_t>(v) * dim + d]);
	            }
	        }
	        const label_t label = local_mesh_->vertex_label(v);
	        if (label != INVALID_LABEL) {
	            kept_vertex_labels.push_back({old2new_vertex[v], label});
	        }
	    }

	    std::vector<CellShape> new_shapes;
	    std::vector<gid_t> new_cell_gids;
	    std::vector<label_t> new_cell_regions;
	    std::vector<size_t> new_cell_refinement_levels;
	    std::vector<offset_t> new_offsets{0};
	    std::vector<index_t> new_conn;

	    new_shapes.reserve(keep_cells.size());
	    new_cell_gids.reserve(keep_cells.size());
	    new_cell_regions.reserve(keep_cells.size());
	    new_cell_refinement_levels.reserve(keep_cells.size());

	    for (index_t c : keep_cells) {
	        new_shapes.push_back(local_mesh_->cell_shape(c));
	        new_cell_gids.push_back(local_mesh_->cell_gids()[static_cast<size_t>(c)]);
	        new_cell_regions.push_back(local_mesh_->cell_region_ids().empty()
	                                       ? 0
	                                       : local_mesh_->cell_region_ids()[static_cast<size_t>(c)]);
	        new_cell_refinement_levels.push_back(local_mesh_->refinement_level(c));

	        auto [verts, n] = local_mesh_->cell_vertices_span(c);
	        for (size_t i = 0; i < n; ++i) {
	            new_conn.push_back(old2new_vertex[verts[i]]);
	        }
        new_offsets.push_back(static_cast<offset_t>(new_conn.size()));
    }

    std::vector<gid_t> new_vertex_gids;
    new_vertex_gids.reserve(keep_vertices.size());
    for (index_t v : keep_vertices) {
        new_vertex_gids.push_back(local_mesh_->vertex_gids()[static_cast<size_t>(v)]);
    }

	    // Preserve vertex and cell fields for kept entities.
	    struct FieldSnapshot {
	        std::string name;
	        EntityKind kind;
	        FieldScalarType type;
        size_t components;
        size_t bytes_per_component;
        bool has_descriptor = false;
        FieldDescriptor descriptor;
        std::shared_ptr<const std::vector<gid_t>> gids; // for face/edge remapping by GID
        std::vector<uint8_t> data;
    };

    std::vector<FieldSnapshot> saved_fields;

	    // Ensure codim-1/codim-2 GIDs are stable before snapshotting face/edge fields.
	    ensure_canonical_face_gids(*local_mesh_);
	    ensure_canonical_edge_gids(*local_mesh_);
	    auto face_gid_snapshot = std::make_shared<std::vector<gid_t>>(local_mesh_->face_gids());
	    auto edge_gid_snapshot = std::make_shared<std::vector<gid_t>>(local_mesh_->edge_gids());
	    const auto old_face_boundary_labels = local_mesh_->face_boundary_ids();
	    const auto old_edge_labels = local_mesh_->edge_label_ids();

	    // Preserve entity sets across the rebuild.
	    struct SetSnapshot {
	        EntityKind kind = EntityKind::Vertex;
	        std::string name;
	        std::vector<index_t> ids;
	        std::vector<gid_t> gids;
	    };
	    std::vector<SetSnapshot> saved_sets;

	    for (const auto& set_name : local_mesh_->list_sets(EntityKind::Vertex)) {
	        const auto& ids = local_mesh_->get_set(EntityKind::Vertex, set_name);
	        std::vector<index_t> members;
	        members.reserve(ids.size());
	        for (const auto v : ids) {
	            const auto it = old2new_vertex.find(v);
	            if (it != old2new_vertex.end()) {
	                members.push_back(it->second);
	            }
	        }
	        if (!members.empty()) {
	            saved_sets.push_back({EntityKind::Vertex, set_name, std::move(members), {}});
	        }
	    }

	    for (const auto& set_name : local_mesh_->list_sets(EntityKind::Volume)) {
	        const auto& ids = local_mesh_->get_set(EntityKind::Volume, set_name);
	        std::vector<index_t> members;
	        members.reserve(ids.size());
	        for (const auto c : ids) {
	            const auto it = old2new_cell.find(c);
	            if (it != old2new_cell.end()) {
	                members.push_back(it->second);
	            }
	        }
	        if (!members.empty()) {
	            saved_sets.push_back({EntityKind::Volume, set_name, std::move(members), {}});
	        }
	    }

	    if (face_gid_snapshot) {
	        const auto& gids = *face_gid_snapshot;
	        for (const auto& set_name : local_mesh_->list_sets(EntityKind::Face)) {
	            const auto& ids = local_mesh_->get_set(EntityKind::Face, set_name);
	            std::vector<gid_t> members;
	            members.reserve(ids.size());
	            for (const auto f : ids) {
	                if (f < 0 || static_cast<size_t>(f) >= gids.size()) continue;
	                const gid_t gid = gids[static_cast<size_t>(f)];
	                if (gid == INVALID_GID) continue;
	                members.push_back(gid);
	            }
	            if (!members.empty()) {
	                saved_sets.push_back({EntityKind::Face, set_name, {}, std::move(members)});
	            }
	        }
	    }

	    if (edge_gid_snapshot) {
	        const auto& gids = *edge_gid_snapshot;
	        for (const auto& set_name : local_mesh_->list_sets(EntityKind::Edge)) {
	            const auto& ids = local_mesh_->get_set(EntityKind::Edge, set_name);
	            std::vector<gid_t> members;
	            members.reserve(ids.size());
	            for (const auto e : ids) {
	                if (e < 0 || static_cast<size_t>(e) >= gids.size()) continue;
	                const gid_t gid = gids[static_cast<size_t>(e)];
	                if (gid == INVALID_GID) continue;
	                members.push_back(gid);
	            }
	            if (!members.empty()) {
	                saved_sets.push_back({EntityKind::Edge, set_name, {}, std::move(members)});
	            }
	        }
	    }

    auto snapshot_fields_subset = [&](EntityKind kind, const std::vector<index_t>& keep_ids) {
        for (const auto& fname : local_mesh_->field_names(kind)) {
            FieldSnapshot snap;
            snap.name = fname;
            snap.kind = kind;
            snap.type = local_mesh_->field_type_by_name(kind, fname);
            snap.components = local_mesh_->field_components_by_name(kind, fname);
            snap.bytes_per_component = local_mesh_->field_bytes_per_component_by_name(kind, fname);

            const auto hnd = local_mesh_->field_handle(kind, fname);
            if (hnd.id != 0) {
                if (const auto* desc = local_mesh_->field_descriptor(hnd)) {
                    snap.has_descriptor = true;
                    snap.descriptor = *desc;
                }
            }

            const uint8_t* src = static_cast<const uint8_t*>(local_mesh_->field_data_by_name(kind, fname));
            if (!src) continue;

            const size_t bpe = snap.components * snap.bytes_per_component;
            snap.data.reserve(keep_ids.size() * bpe);
            for (index_t id : keep_ids) {
                const uint8_t* p = src + static_cast<size_t>(id) * bpe;
                snap.data.insert(snap.data.end(), p, p + bpe);
            }
            saved_fields.push_back(std::move(snap));
        }
    };

    auto snapshot_fields_all = [&](EntityKind kind,
                                   size_t n_entities,
                                   std::shared_ptr<const std::vector<gid_t>> gids) {
        for (const auto& fname : local_mesh_->field_names(kind)) {
            FieldSnapshot snap;
            snap.name = fname;
            snap.kind = kind;
            snap.type = local_mesh_->field_type_by_name(kind, fname);
            snap.components = local_mesh_->field_components_by_name(kind, fname);
            snap.bytes_per_component = local_mesh_->field_bytes_per_component_by_name(kind, fname);
            snap.gids = gids;

            const auto hnd = local_mesh_->field_handle(kind, fname);
            if (hnd.id != 0) {
                if (const auto* desc = local_mesh_->field_descriptor(hnd)) {
                    snap.has_descriptor = true;
                    snap.descriptor = *desc;
                }
            }

            const uint8_t* src = static_cast<const uint8_t*>(local_mesh_->field_data_by_name(kind, fname));
            if (!src) continue;

            const size_t bpe = snap.components * snap.bytes_per_component;
            snap.data.assign(src, src + n_entities * bpe);
            saved_fields.push_back(std::move(snap));
        }
    };

    snapshot_fields_subset(EntityKind::Vertex, keep_vertices);
    snapshot_fields_subset(EntityKind::Volume, keep_cells);
    snapshot_fields_all(EntityKind::Face, local_mesh_->n_faces(), face_gid_snapshot);
    snapshot_fields_all(EntityKind::Edge, local_mesh_->n_edges(), edge_gid_snapshot);

    local_mesh_->clear();
    local_mesh_->build_from_arrays(dim, new_coords, new_offsets, new_conn, new_shapes);
    local_mesh_->set_vertex_gids(std::move(new_vertex_gids));
	    local_mesh_->set_cell_gids(std::move(new_cell_gids));
	    local_mesh_->finalize();

	    ensure_canonical_face_gids(*local_mesh_);
	    ensure_canonical_edge_gids(*local_mesh_);

	    // Restore label registry (name <-> id).
	    for (const auto& [label, name] : label_registry) {
	        local_mesh_->register_label(name, label);
	    }

	    // Restore current coordinates and active configuration.
	    if (old_has_current_coords) {
	        local_mesh_->set_current_coords(new_current_coords);
	    }
	    if (old_active_config == Configuration::Current && old_has_current_coords) {
	        local_mesh_->use_current_configuration();
	    } else {
	        local_mesh_->use_reference_configuration();
	    }

	    // Restore vertex labels.
	    for (const auto& [v, label] : kept_vertex_labels) {
	        local_mesh_->set_vertex_label(v, label);
	    }

	    // Restore cell region labels.
	    for (index_t c = 0; c < static_cast<index_t>(new_cell_regions.size()); ++c) {
	        local_mesh_->set_region_label(c, new_cell_regions[static_cast<size_t>(c)]);
	    }

	    // Restore cell refinement levels.
	    if (!new_cell_refinement_levels.empty() && new_cell_refinement_levels.size() == local_mesh_->n_cells()) {
	        local_mesh_->set_cell_refinement_levels(new_cell_refinement_levels);
	    }

	    // Restore face boundary labels and edge labels by canonical GID.
	    if (face_gid_snapshot && !old_face_boundary_labels.empty()) {
	        const auto& gids = *face_gid_snapshot;
	        const size_t n = std::min(gids.size(), old_face_boundary_labels.size());
	        for (size_t i = 0; i < n; ++i) {
	            const label_t label = old_face_boundary_labels[i];
	            if (label == INVALID_LABEL) continue;
	            const gid_t gid = gids[i];
	            if (gid == INVALID_GID) continue;
	            const index_t f = local_mesh_->global_to_local_face(gid);
	            if (f == INVALID_INDEX) continue;
	            local_mesh_->set_boundary_label(f, label);
	        }
	    }

	    if (edge_gid_snapshot && !old_edge_labels.empty()) {
	        const auto& gids = *edge_gid_snapshot;
	        const size_t n = std::min(gids.size(), old_edge_labels.size());
	        for (size_t i = 0; i < n; ++i) {
	            const label_t label = old_edge_labels[i];
	            if (label == INVALID_LABEL) continue;
	            const gid_t gid = gids[i];
	            if (gid == INVALID_GID) continue;
	            const index_t e = local_mesh_->global_to_local_edge(gid);
	            if (e == INVALID_INDEX) continue;
	            local_mesh_->set_edge_label(e, label);
	        }
	    }

	    // Restore entity sets.
	    for (const auto& set : saved_sets) {
	        if (set.kind == EntityKind::Vertex || set.kind == EntityKind::Volume) {
	            for (const auto id : set.ids) {
	                local_mesh_->add_to_set(set.kind, set.name, id);
	            }
	        } else if (set.kind == EntityKind::Face) {
	            for (const auto gid : set.gids) {
	                const index_t f = local_mesh_->global_to_local_face(gid);
	                if (f == INVALID_INDEX) continue;
	                local_mesh_->add_to_set(EntityKind::Face, set.name, f);
	            }
	        } else if (set.kind == EntityKind::Edge) {
	            for (const auto gid : set.gids) {
	                const index_t e = local_mesh_->global_to_local_edge(gid);
	                if (e == INVALID_INDEX) continue;
	                local_mesh_->add_to_set(EntityKind::Edge, set.name, e);
	            }
	        }
	    }

	    // Restore fields (all kinds), remapping face/edge data by canonical GIDs.
	    for (const auto& snap : saved_fields) {
	        auto h = local_mesh_->attach_field(snap.kind, snap.name, snap.type, snap.components, snap.bytes_per_component);
	        uint8_t* dst = static_cast<uint8_t*>(local_mesh_->field_data(h));
        if (!dst) continue;

        const size_t bpe = snap.components * snap.bytes_per_component;
        const size_t new_count = local_mesh_->field_entity_count(h);
        std::memset(dst, 0, new_count * bpe);

        if (snap.kind == EntityKind::Vertex || snap.kind == EntityKind::Volume) {
            const size_t n_copy = std::min(snap.data.size(), new_count * bpe);
            std::memcpy(dst, snap.data.data(), n_copy);
        } else if (snap.kind == EntityKind::Face || snap.kind == EntityKind::Edge) {
            if (!snap.gids) continue;
            const auto& gids = *snap.gids;
            const size_t n_old = std::min(gids.size(), snap.data.size() / bpe);
            for (size_t i = 0; i < n_old; ++i) {
                const gid_t gid = gids[i];
                const index_t new_id =
                    (snap.kind == EntityKind::Face)
                        ? local_mesh_->global_to_local_face(gid)
                        : local_mesh_->global_to_local_edge(gid);
                if (new_id == INVALID_INDEX || new_id < 0 || static_cast<size_t>(new_id) >= new_count) {
                    continue;
                }
                std::memcpy(dst + static_cast<size_t>(new_id) * bpe,
                            snap.data.data() + i * bpe,
                            bpe);
            }
        }

        if (snap.has_descriptor) {
            local_mesh_->set_field_descriptor(h, snap.descriptor);
        }
    }

    // Reset ownership for base mesh and recompute shared entities.
    vertex_owner_.assign(local_mesh_->n_vertices(), Ownership::Owned);
    vertex_owner_rank_.assign(local_mesh_->n_vertices(), my_rank_);
    cell_owner_.assign(local_mesh_->n_cells(), Ownership::Owned);
    cell_owner_rank_.assign(local_mesh_->n_cells(), my_rank_);
    face_owner_.assign(local_mesh_->n_faces(), Ownership::Owned);
    face_owner_rank_.assign(local_mesh_->n_faces(), my_rank_);
    edge_owner_.assign(local_mesh_->n_edges(), Ownership::Owned);
    edge_owner_rank_.assign(local_mesh_->n_edges(), my_rank_);

    gather_shared_entities();
    notify_partition_changed();
#endif
}

void DistributedMesh::update_ghosts(const std::vector<FieldHandle>& fields) {
    // =====================================================
    // Ghost Field Update Algorithm
    // =====================================================
    // Purpose: Update ghost entity field values from their owners
    // Algorithm: Use pre-built exchange patterns for efficient communication
    //
    // Performance characteristics:
    // - Time: O(n_fields * (n_ghosts * field_size + comm_latency))
    // - Memory: O(max(send_buffer_size, recv_buffer_size))
    // - Communication: 2 * n_neighbors messages per field (non-blocking)
    // - Scaling: O(log P) neighbors typical, O(√P) worst case
    //
    // Optimizations:
    // - Uses pre-built exchange patterns (avoids runtime construction)
    // - Non-blocking communication overlaps computation
    // - Batching multiple fields would reduce latency overhead

#ifdef MESH_HAS_MPI
    if (world_size_ == 1) {
        return;  // No ghosts/shared entities in serial
    }
    if (!local_mesh_) {
        return;
    }
    if (fields.empty()) {
        return;
    }

    // Check if exchange patterns are built
    if (vertex_exchange_.send_ranks.empty() &&
        vertex_exchange_.recv_ranks.empty() &&
        cell_exchange_.send_ranks.empty() &&
        cell_exchange_.recv_ranks.empty() &&
        face_exchange_.send_ranks.empty() &&
        face_exchange_.recv_ranks.empty() &&
        edge_exchange_.send_ranks.empty() &&
        edge_exchange_.recv_ranks.empty()) {
        // Patterns not built - build them now
        build_exchange_patterns();
    }

    auto pattern_for = [&](EntityKind kind) -> const ExchangePattern& {
        switch (kind) {
            case EntityKind::Vertex: return vertex_exchange_;
            case EntityKind::Volume: return cell_exchange_;
            case EntityKind::Face: return face_exchange_;
            case EntityKind::Edge: return edge_exchange_;
        }
        return vertex_exchange_;
    };

    auto tag_for = [&](EntityKind kind, int phase) -> int {
        // phase 0: Exchange, 1: Accumulate reduce-to-owner, 2: Accumulate broadcast-to-copies
        return 100 + phase * 10 + static_cast<int>(kind);
    };

    auto exchange_fields_packed = [&](EntityKind kind,
                                      const std::vector<FieldHandle>& group,
                                      int mpi_tag) {
        if (group.empty()) return;
        const auto& pattern = pattern_for(kind);
        if (pattern.send_ranks.empty() && pattern.recv_ranks.empty()) return;

        size_t n_entities = 0;
        switch (kind) {
            case EntityKind::Vertex: n_entities = local_mesh_->n_vertices(); break;
            case EntityKind::Volume: n_entities = local_mesh_->n_cells(); break;
            case EntityKind::Face: n_entities = local_mesh_->n_faces(); break;
            case EntityKind::Edge: n_entities = local_mesh_->n_edges(); break;
        }

        struct FieldInfo {
            FieldHandle h;
            std::uint8_t* data = nullptr;
            size_t bytes_per_entity = 0;
            size_t offset = 0;
        };

        std::vector<FieldInfo> infos;
        infos.reserve(group.size());

        size_t record_bytes = 0;
        for (const auto& h : group) {
            auto* data = static_cast<std::uint8_t*>(local_mesh_->field_data(h));
            if (!data) {
                continue;
            }
            const size_t bpe = local_mesh_->field_bytes_per_entity(h);
            FieldInfo info;
            info.h = h;
            info.data = data;
            info.bytes_per_entity = bpe;
            info.offset = record_bytes;
            record_bytes += bpe;
            infos.push_back(info);
        }
        if (infos.empty() || record_bytes == 0) {
            return;
        }

        auto pack_records = [&](const std::vector<index_t>& entities,
                                std::vector<std::uint8_t>& buffer) {
            buffer.resize(entities.size() * record_bytes);
            for (size_t j = 0; j < entities.size(); ++j) {
                const index_t entity = entities[j];
                std::uint8_t* out = buffer.data() + j * record_bytes;
                for (const auto& f : infos) {
                    std::uint8_t* dst = out + f.offset;
                    if (entity < 0 || static_cast<size_t>(entity) >= n_entities) {
                        std::memset(dst, 0, f.bytes_per_entity);
                        continue;
                    }
                    std::memcpy(dst, f.data + static_cast<size_t>(entity) * f.bytes_per_entity, f.bytes_per_entity);
                }
            }
        };

        // Allocate buffers
        std::vector<std::vector<std::uint8_t>> send_buffers(pattern.send_ranks.size());
        std::vector<std::vector<std::uint8_t>> recv_buffers(pattern.recv_ranks.size());

        // Pack send buffers
        for (size_t i = 0; i < pattern.send_ranks.size(); ++i) {
            pack_records(pattern.send_lists[i], send_buffers[i]);
        }

        // Size recv buffers
        for (size_t i = 0; i < pattern.recv_ranks.size(); ++i) {
            recv_buffers[i].resize(pattern.recv_lists[i].size() * record_bytes);
        }

        std::vector<MPI_Request> requests;
        requests.reserve(pattern.send_ranks.size() + pattern.recv_ranks.size());

        // Non-blocking sends
        for (size_t i = 0; i < pattern.send_ranks.size(); ++i) {
            if (send_buffers[i].empty()) {
                continue;
            }
            MPI_Request req;
            MPI_Isend(send_buffers[i].data(),
                      static_cast<int>(send_buffers[i].size()),
                      MPI_BYTE, pattern.send_ranks[i], mpi_tag, comm_, &req);
            requests.push_back(req);
        }

        // Non-blocking receives
        for (size_t i = 0; i < pattern.recv_ranks.size(); ++i) {
            if (recv_buffers[i].empty()) {
                continue;
            }
            MPI_Request req;
            MPI_Irecv(recv_buffers[i].data(),
                      static_cast<int>(recv_buffers[i].size()),
                      MPI_BYTE, pattern.recv_ranks[i], mpi_tag, comm_, &req);
            requests.push_back(req);
        }

        if (!requests.empty()) {
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        }

        // Unpack recv buffers (overwrite)
        for (size_t i = 0; i < pattern.recv_ranks.size(); ++i) {
            const auto& recv_list = pattern.recv_lists[i];
            const auto& buf = recv_buffers[i];
            if (recv_list.empty() || buf.empty()) continue;

            for (size_t j = 0; j < recv_list.size(); ++j) {
                const index_t entity = recv_list[j];
                if (entity < 0 || static_cast<size_t>(entity) >= n_entities) continue;
                const std::uint8_t* in = buf.data() + j * record_bytes;
                for (const auto& f : infos) {
                    std::memcpy(f.data + static_cast<size_t>(entity) * f.bytes_per_entity,
                                in + f.offset,
                                f.bytes_per_entity);
                }
            }
        }
    };

    auto accumulate_fields_typed = [&](EntityKind kind,
                                       FieldScalarType scalar_type,
                                       const std::vector<FieldHandle>& group,
                                       int reduce_tag,
                                       int bcast_tag) {
        if (group.empty()) return;
        const auto& pattern = pattern_for(kind);
        if (pattern.send_ranks.empty() && pattern.recv_ranks.empty()) return;

        size_t n_entities = 0;
        switch (kind) {
            case EntityKind::Vertex: n_entities = local_mesh_->n_vertices(); break;
            case EntityKind::Volume: n_entities = local_mesh_->n_cells(); break;
            case EntityKind::Face: n_entities = local_mesh_->n_faces(); break;
            case EntityKind::Edge: n_entities = local_mesh_->n_edges(); break;
        }

        struct TypedField {
            FieldHandle h;
            size_t components = 0;
            size_t comp_offset = 0;
            void* data = nullptr;
        };

        std::vector<TypedField> flds;
        flds.reserve(group.size());

        size_t total_components = 0;
        for (const auto& h : group) {
            if (local_mesh_->field_type(h) != scalar_type) {
                continue;
            }
            const size_t comps = local_mesh_->field_components(h);
            const size_t bpe = local_mesh_->field_bytes_per_entity(h);
            const size_t bpc = (comps == 0) ? 0 : (bpe / comps);
            if (comps == 0 || bpc * comps != bpe) {
                throw std::runtime_error(
                    "DistributedMesh::update_ghosts: invalid field bytes-per-entity/components relationship");
            }
            if (bytes_per(scalar_type) == 0 || bpc != bytes_per(scalar_type)) {
                throw std::runtime_error(
                    "DistributedMesh::update_ghosts: Accumulate policy requires a built-in scalar type");
            }
            void* data = local_mesh_->field_data(h);
            if (!data) continue;
            TypedField tf;
            tf.h = h;
            tf.components = comps;
            tf.comp_offset = total_components;
            tf.data = data;
            total_components += comps;
            flds.push_back(tf);
        }
        if (flds.empty() || total_components == 0) {
            return;
        }

        // Typed reduction (ghost/shared -> owner, sum)
        auto reduce_impl = [&](auto dummy) {
            using T = decltype(dummy);
            // Resolve typed pointers for all fields.
            std::vector<T*> ptrs;
            ptrs.reserve(flds.size());
            for (const auto& f : flds) {
                ptrs.push_back(static_cast<T*>(f.data));
            }

            auto pack_typed = [&](const std::vector<index_t>& entities, std::vector<T>& buffer) {
                buffer.assign(entities.size() * total_components, T{});
                for (size_t j = 0; j < entities.size(); ++j) {
                    const index_t entity = entities[j];
                    if (entity < 0 || static_cast<size_t>(entity) >= n_entities) continue;
                    T* out = buffer.data() + j * total_components;
                    for (size_t fi = 0; fi < flds.size(); ++fi) {
                        const auto& f = flds[fi];
                        const T* src = ptrs[fi] + static_cast<size_t>(entity) * f.components;
                        std::memcpy(out + f.comp_offset, src, f.components * sizeof(T));
                    }
                }
            };

            std::vector<std::vector<T>> send_buffers(pattern.recv_ranks.size());
            std::vector<std::vector<T>> recv_buffers(pattern.send_ranks.size());

            // Send contributions for non-owned entities back to their owners (pattern.recv_*).
            for (size_t i = 0; i < pattern.recv_ranks.size(); ++i) {
                pack_typed(pattern.recv_lists[i], send_buffers[i]);
            }

            // Receive contributions for owned entities from ghost/shared copies (pattern.send_*).
            for (size_t i = 0; i < pattern.send_ranks.size(); ++i) {
                recv_buffers[i].assign(pattern.send_lists[i].size() * total_components, T{});
            }

            std::vector<MPI_Request> requests;
            requests.reserve(pattern.recv_ranks.size() + pattern.send_ranks.size());

            for (size_t i = 0; i < pattern.recv_ranks.size(); ++i) {
                if (send_buffers[i].empty()) continue;
                MPI_Request req;
                MPI_Isend(reinterpret_cast<const void*>(send_buffers[i].data()),
                          static_cast<int>(send_buffers[i].size() * sizeof(T)),
                          MPI_BYTE, pattern.recv_ranks[i], reduce_tag, comm_, &req);
                requests.push_back(req);
            }

            for (size_t i = 0; i < pattern.send_ranks.size(); ++i) {
                if (recv_buffers[i].empty()) continue;
                MPI_Request req;
                MPI_Irecv(reinterpret_cast<void*>(recv_buffers[i].data()),
                          static_cast<int>(recv_buffers[i].size() * sizeof(T)),
                          MPI_BYTE, pattern.send_ranks[i], reduce_tag, comm_, &req);
                requests.push_back(req);
            }

            if (!requests.empty()) {
                MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            }

            // Unpack and add into owned entities.
            for (size_t i = 0; i < pattern.send_ranks.size(); ++i) {
                const auto& recv_list = pattern.send_lists[i];
                const auto& buf = recv_buffers[i];
                if (recv_list.empty() || buf.empty()) continue;

                for (size_t j = 0; j < recv_list.size(); ++j) {
                    const index_t entity = recv_list[j];
                    if (entity < 0 || static_cast<size_t>(entity) >= n_entities) continue;
                    const T* in = buf.data() + j * total_components;
                    for (size_t fi = 0; fi < flds.size(); ++fi) {
                        const auto& f = flds[fi];
                        T* dst = ptrs[fi] + static_cast<size_t>(entity) * f.components;
                        const T* src = in + f.comp_offset;
                        for (size_t c = 0; c < f.components; ++c) {
                            dst[c] += src[c];
                        }
                    }
                }
            }
        };

        switch (scalar_type) {
            case FieldScalarType::Float64:
                reduce_impl(static_cast<double>(0));
                break;
            case FieldScalarType::Float32:
                reduce_impl(static_cast<float>(0));
                break;
            case FieldScalarType::Int32:
                reduce_impl(static_cast<std::int32_t>(0));
                break;
            case FieldScalarType::Int64:
                reduce_impl(static_cast<std::int64_t>(0));
                break;
            case FieldScalarType::UInt8:
            case FieldScalarType::Custom:
                throw std::runtime_error(
                    "DistributedMesh::update_ghosts: Accumulate policy is not supported for this scalar type");
        }

        // Broadcast accumulated owner values back to all non-owned copies so all ranks are consistent.
        exchange_fields_packed(kind, group, bcast_tag);
    };

    // Group fields by ghost policy + kind for efficient batching.
    std::array<std::vector<FieldHandle>, 4> exchange_groups;
    std::map<std::pair<int, int>, std::vector<FieldHandle>> accumulate_groups;

    auto kind_index = [&](EntityKind kind) -> size_t {
        switch (kind) {
            case EntityKind::Vertex: return 0;
            case EntityKind::Volume: return 1;
            case EntityKind::Face: return 2;
            case EntityKind::Edge: return 3;
        }
        return 0;
    };

    for (const auto& h : fields) {
        if (h.id == 0) continue;
        if (!local_mesh_->field_data(h)) continue;

        FieldGhostPolicy policy = FieldGhostPolicy::Exchange;
        if (const auto* desc = local_mesh_->field_descriptor(h)) {
            policy = desc->ghost_policy;
        }
        if (policy == FieldGhostPolicy::None) {
            continue;
        }

        if (policy == FieldGhostPolicy::Exchange) {
            exchange_groups[kind_index(h.kind)].push_back(h);
        } else if (policy == FieldGhostPolicy::Accumulate) {
            const auto type = local_mesh_->field_type(h);
            accumulate_groups[{static_cast<int>(h.kind), static_cast<int>(type)}].push_back(h);
        }
    }

    // Performance timing (optional)
    double update_start = MPI_Wtime();

    // Exchange-only fields (owner -> copies).
    exchange_fields_packed(EntityKind::Vertex, exchange_groups[0], tag_for(EntityKind::Vertex, 0));
    exchange_fields_packed(EntityKind::Volume, exchange_groups[1], tag_for(EntityKind::Volume, 0));
    exchange_fields_packed(EntityKind::Face, exchange_groups[2], tag_for(EntityKind::Face, 0));
    exchange_fields_packed(EntityKind::Edge, exchange_groups[3], tag_for(EntityKind::Edge, 0));

    // Accumulate fields (reduce to owner, then broadcast to copies).
    for (const auto& [key, group] : accumulate_groups) {
        const EntityKind kind = static_cast<EntityKind>(key.first);
        const FieldScalarType type = static_cast<FieldScalarType>(key.second);
        accumulate_fields_typed(kind,
                                type,
                                group,
                                tag_for(kind, 1),
                                tag_for(kind, 2));
    }

    if (getenv("MESH_VERBOSE")) {
        double update_time = MPI_Wtime() - update_start;
        double max_update_time = 0.0;
        MPI_Reduce(&update_time, &max_update_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);

        if (my_rank_ == 0) {
            std::cout << "Ghost/shared update for " << fields.size()
                      << " fields completed in " << max_update_time << " seconds\n";
        }
    }

    local_mesh_->event_bus().notify(MeshEvent::FieldsChanged);
	#endif
	}

void DistributedMesh::update_exchange_ghost_fields() {
#ifdef MESH_HAS_MPI
    if (!local_mesh_) {
        return;
    }
    const auto fields = MeshFields::fields_requiring_ghost_update(*local_mesh_);
    update_ghosts(fields);
#endif
}

void DistributedMesh::update_exchange_ghost_coordinates(Configuration cfg) {
#ifdef MESH_HAS_MPI
    if (!local_mesh_) {
        return;
    }
    if (world_size_ == 1) {
        return;
    }

    const bool use_current = (cfg == Configuration::Current || cfg == Configuration::Deformed);
    if (!use_current) {
        return;
    }

    // Ensure current coordinates exist for in-place exchange.
    if (!local_mesh_->has_current_coords()) {
        local_mesh_->set_current_coords(local_mesh_->X_ref());
    }

    const int dim = local_mesh_->dim();
    if (dim <= 0) {
        return;
    }

    // Check if exchange patterns are built.
    if (vertex_exchange_.send_ranks.empty() && vertex_exchange_.recv_ranks.empty() &&
        cell_exchange_.send_ranks.empty() && cell_exchange_.recv_ranks.empty() &&
        face_exchange_.send_ranks.empty() && face_exchange_.recv_ranks.empty()) {
        build_exchange_patterns();
    }

    auto* coords = local_mesh_->X_cur_data_mutable();
    if (!coords) {
        return;
    }

    const size_t bytes_per_vertex = static_cast<size_t>(dim) * sizeof(real_t);
    exchange_entity_data(EntityKind::Vertex,
                         coords,
                         coords,
                         bytes_per_vertex,
                         vertex_exchange_);

    local_mesh_->event_bus().notify(MeshEvent::GeometryChanged);
#else
    (void)cfg;
#endif
}

// ==========================================
// Migration & Load Balancing
// ==========================================

	void DistributedMesh::migrate(const std::vector<rank_t>& new_owner_rank_per_cell) {
	#ifdef MESH_HAS_MPI
	    if (world_size_ == 1) {
	        return;
	    }

	    if (new_owner_rank_per_cell.size() != local_mesh_->n_cells()) {
	        throw std::invalid_argument("migrate: wrong size for new_owner_rank_per_cell");
	    }

	    const Configuration old_active_config = local_mesh_->active_configuration();
	    const bool old_has_current_coords = local_mesh_->has_current_coords();

	    // ========================================
	    // Step 1: Determine migration destinations
	    // ========================================

	    std::map<rank_t, std::vector<index_t>> cells_to_send;
	    std::vector<index_t> cells_to_keep;

	    auto is_locally_owned_cell = [&](index_t c) -> bool {
	        if (c < 0) return false;
	        if (static_cast<size_t>(c) >= local_mesh_->n_cells()) return false;
	        if (cell_owner_rank_.size() == local_mesh_->n_cells()) {
	            if (cell_owner_.size() == local_mesh_->n_cells() &&
	                cell_owner_[static_cast<size_t>(c)] == Ownership::Ghost) {
	                return false;
	            }
	            return cell_owner_rank_[static_cast<size_t>(c)] == my_rank_;
	        }
	        return true;
	    };

	    for (index_t c = 0; c < static_cast<index_t>(new_owner_rank_per_cell.size()); ++c) {
	        if (!is_locally_owned_cell(c)) {
	            continue;
	        }
	        rank_t new_rank = new_owner_rank_per_cell[c];
	        if (new_rank == my_rank_) {
	            cells_to_keep.push_back(c);
	        } else if (new_rank >= 0 && new_rank < world_size_) {
	            cells_to_send[new_rank].push_back(c);
	        }
	    }

    // Preserve label registry (name <-> id) across rebuild.
    const auto label_registry = local_mesh_->list_label_names();

    // Ensure codimension IDs are stable before packing face/edge metadata.
    ensure_canonical_face_gids(*local_mesh_);
    ensure_canonical_edge_gids(*local_mesh_);

    // ========================================
    // Step 3: Pack data into send buffers
    // ========================================

	    struct MeshPacket {
	        enum : std::uint32_t {
	            kMagic = 0x53564d50u, // 'SVMP'
	            kVersion = 3u,
	        };

	        // Header information
	        std::uint64_t n_cells = 0;
	        std::uint64_t n_vertices = 0;
	        std::uint64_t n_faces = 0;
	        std::uint64_t n_edges = 0;
	        int spatial_dim = 0;
	        int has_current_coords = 0;

	        // Vertex data
	        std::vector<gid_t> vertex_gids;
	        std::vector<real_t> vertex_coords_ref;
	        std::vector<real_t> vertex_coords_cur;
	        std::vector<label_t> vertex_labels;

        // Cell data
        std::vector<gid_t> cell_gids;
        std::vector<CellShape> cell_shapes;
        std::vector<offset_t> cell_offsets;
        std::vector<index_t> cell_connectivity;
        std::vector<label_t> cell_regions;
        std::vector<std::uint64_t> cell_refinement_levels;

        // Codim-1/codim-2 canonical GIDs for remapping metadata and fields.
        std::vector<gid_t> face_gids;
        std::vector<gid_t> edge_gids;

        // Sparse labels keyed by canonical GID.
        std::vector<std::pair<gid_t, label_t>> face_boundary_labels;
        std::vector<std::pair<gid_t, label_t>> edge_labels;

        struct SetPacket {
            EntityKind kind = EntityKind::Vertex;
            std::string name;
            std::vector<gid_t> gids; // vertex/cell gids, or canonical face/edge gids
        };
        std::vector<SetPacket> sets;

        // Field data (stores field bytes + optional descriptor).
        struct FieldPacket {
            std::string name;
            EntityKind kind;
            FieldScalarType type;
            std::uint64_t components = 0;
            std::uint64_t bytes_per_component = 0;
            bool has_descriptor = false;
            FieldDescriptor descriptor;
            std::vector<uint8_t> data;
        };
        std::vector<FieldPacket> fields;

        size_t compute_size() const {
            auto string_size = [](const std::string& s) -> size_t {
                return sizeof(std::uint64_t) + s.size();
            };

            auto descriptor_size = [&](const FieldDescriptor& desc) -> size_t {
                size_t size = 0;
                size += sizeof(int);                // location
                size += sizeof(std::uint64_t);      // components
                size += sizeof(real_t);             // unit_scale
                size += sizeof(int);                // time_dependent
                size += sizeof(int);                // intent
                size += sizeof(int);                // ghost_policy
                size += string_size(desc.units);
                size += string_size(desc.description);
                size += sizeof(std::uint64_t);      // n_component_names
                for (const auto& n : desc.component_names) {
                    size += string_size(n);
                }
                return size;
            };

            size_t size = 0;
	            size += sizeof(std::uint32_t) * 2; // magic + version
	            size += sizeof(std::uint64_t) * 4; // n_cells, n_vertices, n_faces, n_edges
	            size += sizeof(int);               // spatial_dim
	            size += sizeof(int);               // has_current_coords

	            size += vertex_gids.size() * sizeof(gid_t);
	            size += vertex_coords_ref.size() * sizeof(real_t);
	            if (has_current_coords) {
	                size += vertex_coords_cur.size() * sizeof(real_t);
	            }
	            size += vertex_labels.size() * sizeof(label_t);

            size += cell_gids.size() * sizeof(gid_t);
            size += cell_shapes.size() * sizeof(CellShape);
            size += cell_offsets.size() * sizeof(offset_t);
            size += cell_connectivity.size() * sizeof(index_t);
            size += cell_regions.size() * sizeof(label_t);
            size += cell_refinement_levels.size() * sizeof(std::uint64_t);

            size += face_gids.size() * sizeof(gid_t);
            size += edge_gids.size() * sizeof(gid_t);

            size += sizeof(std::uint64_t); // face_boundary_labels count
            size += face_boundary_labels.size() * (sizeof(gid_t) + sizeof(label_t));
            size += sizeof(std::uint64_t); // edge_labels count
            size += edge_labels.size() * (sizeof(gid_t) + sizeof(label_t));

            size += sizeof(std::uint64_t); // sets count
            for (const auto& set : sets) {
                size += sizeof(int); // kind
                size += string_size(set.name);
                size += sizeof(std::uint64_t); // n_ids
                size += set.gids.size() * sizeof(gid_t);
            }

            size += sizeof(std::uint64_t); // fields count
            for (const auto& field : fields) {
                size += sizeof(int) * 2; // kind + type
                size += sizeof(std::uint64_t) * 2; // components + bytes_per_component
                size += string_size(field.name);
                size += sizeof(int); // has_descriptor
                if (field.has_descriptor) {
                    size += descriptor_size(field.descriptor);
                }
                size += sizeof(std::uint64_t); // data size
                size += field.data.size();
            }

            return size;
        }

        void serialize(std::vector<uint8_t>& buffer) const {
            buffer.clear();
            buffer.reserve(compute_size());

            // Helper to append data to buffer
            auto append = [&buffer](const void* data, size_t size) {
                if (size == 0) return;
                const uint8_t* bytes = static_cast<const uint8_t*>(data);
                buffer.insert(buffer.end(), bytes, bytes + size);
            };

            auto append_u32 = [&](std::uint32_t v) { append(&v, sizeof(std::uint32_t)); };
            auto append_u64 = [&](std::uint64_t v) { append(&v, sizeof(std::uint64_t)); };
            auto append_i32 = [&](int v) { append(&v, sizeof(int)); };

            auto append_string = [&](const std::string& s) {
                const std::uint64_t len = static_cast<std::uint64_t>(s.size());
                append_u64(len);
                if (len > 0) {
                    append(s.data(), static_cast<size_t>(len));
                }
            };

            auto append_field_descriptor = [&](const FieldDescriptor& desc) {
                const int location = static_cast<int>(desc.location);
                const std::uint64_t components = static_cast<std::uint64_t>(desc.components);
                const real_t unit_scale = desc.unit_scale;
                const int time_dep = desc.time_dependent ? 1 : 0;
                const int intent = static_cast<int>(desc.intent);
                const int ghost_policy = static_cast<int>(desc.ghost_policy);

                append_i32(location);
                append_u64(components);
                append(&unit_scale, sizeof(real_t));
                append_i32(time_dep);
                append_i32(intent);
                append_i32(ghost_policy);

                append_string(desc.units);
                append_string(desc.description);

                const std::uint64_t n_comp_names = static_cast<std::uint64_t>(desc.component_names.size());
                append_u64(n_comp_names);
                for (const auto& n : desc.component_names) {
                    append_string(n);
                }
            };

            // Serialize header
            append_u32(kMagic);
            append_u32(kVersion);
	            append_u64(n_cells);
	            append_u64(n_vertices);
	            append_u64(n_faces);
	            append_u64(n_edges);
	            append_i32(spatial_dim);
	            append_i32(has_current_coords);

	            // Serialize header
	            // Serialize vertices
	            if (!vertex_gids.empty()) {
	                append(vertex_gids.data(), vertex_gids.size() * sizeof(gid_t));
	            }
	            if (!vertex_coords_ref.empty()) {
	                append(vertex_coords_ref.data(), vertex_coords_ref.size() * sizeof(real_t));
	            }
	            if (!vertex_labels.empty()) {
	                append(vertex_labels.data(), vertex_labels.size() * sizeof(label_t));
	            }
	            if (has_current_coords && !vertex_coords_cur.empty()) {
	                append(vertex_coords_cur.data(), vertex_coords_cur.size() * sizeof(real_t));
	            }

            // Serialize cells
            if (!cell_gids.empty()) {
                append(cell_gids.data(), cell_gids.size() * sizeof(gid_t));
            }
            if (!cell_shapes.empty()) {
                append(cell_shapes.data(), cell_shapes.size() * sizeof(CellShape));
            }
            if (!cell_offsets.empty()) {
                append(cell_offsets.data(), cell_offsets.size() * sizeof(offset_t));
            }
            if (!cell_connectivity.empty()) {
                append(cell_connectivity.data(), cell_connectivity.size() * sizeof(index_t));
            }
            if (!cell_regions.empty()) {
                append(cell_regions.data(), cell_regions.size() * sizeof(label_t));
            }
            if (!cell_refinement_levels.empty()) {
                append(cell_refinement_levels.data(),
                       cell_refinement_levels.size() * sizeof(std::uint64_t));
            }

            // Serialize canonical face/edge gids
            if (!face_gids.empty()) {
                append(face_gids.data(), face_gids.size() * sizeof(gid_t));
            }
            if (!edge_gids.empty()) {
                append(edge_gids.data(), edge_gids.size() * sizeof(gid_t));
            }

            // Serialize sparse labels
            append_u64(static_cast<std::uint64_t>(face_boundary_labels.size()));
            for (const auto& [gid, label] : face_boundary_labels) {
                append(&gid, sizeof(gid_t));
                append(&label, sizeof(label_t));
            }

            append_u64(static_cast<std::uint64_t>(edge_labels.size()));
            for (const auto& [gid, label] : edge_labels) {
                append(&gid, sizeof(gid_t));
                append(&label, sizeof(label_t));
            }

            // Serialize sets
            append_u64(static_cast<std::uint64_t>(sets.size()));
            for (const auto& set : sets) {
                append_i32(static_cast<int>(set.kind));
                append_string(set.name);
                append_u64(static_cast<std::uint64_t>(set.gids.size()));
                if (!set.gids.empty()) {
                    append(set.gids.data(), set.gids.size() * sizeof(gid_t));
                }
            }

            // Serialize fields
            append_u64(static_cast<std::uint64_t>(fields.size()));
            for (const auto& field : fields) {
                append_i32(static_cast<int>(field.kind));
                append_i32(static_cast<int>(field.type));
                append_u64(field.components);
                append_u64(field.bytes_per_component);
                append_string(field.name);

                const int has_desc = field.has_descriptor ? 1 : 0;
                append_i32(has_desc);
                if (field.has_descriptor) {
                    append_field_descriptor(field.descriptor);
                }

                append_u64(static_cast<std::uint64_t>(field.data.size()));
                if (!field.data.empty()) {
                    append(field.data.data(), field.data.size());
                }
            }
        }

        void deserialize(const std::vector<uint8_t>& buffer) {
            size_t offset = 0;

            // Helper to read data from buffer
            auto read = [&buffer, &offset](void* dest, size_t size) {
                if (offset + size > buffer.size()) {
                    throw std::runtime_error("MeshPacket::deserialize: truncated buffer");
                }
                std::memcpy(dest, &buffer[offset], size);
                offset += size;
            };

            auto read_u32 = [&]() -> std::uint32_t {
                std::uint32_t v = 0;
                read(&v, sizeof(std::uint32_t));
                return v;
            };
            auto read_u64 = [&]() -> std::uint64_t {
                std::uint64_t v = 0;
                read(&v, sizeof(std::uint64_t));
                return v;
            };
            auto read_i32 = [&]() -> int {
                int v = 0;
                read(&v, sizeof(int));
                return v;
            };

            auto read_string = [&]() -> std::string {
                const std::uint64_t len = read_u64();
                std::string s;
                s.resize(static_cast<size_t>(len));
                if (len > 0) {
                    read(s.data(), static_cast<size_t>(len));
                }
                return s;
            };

            auto read_field_descriptor = [&]() -> FieldDescriptor {
                FieldDescriptor desc;
                const int location = read_i32();
                const std::uint64_t comps = read_u64();
                real_t unit_scale = 1.0;
                read(&unit_scale, sizeof(real_t));
                const int time_dep = read_i32();
                const int intent = read_i32();
                const int ghost_policy = read_i32();

                desc.location = static_cast<EntityKind>(location);
                desc.components = static_cast<size_t>(comps);
                desc.unit_scale = unit_scale;
                desc.time_dependent = (time_dep != 0);
                desc.intent = static_cast<FieldIntent>(intent);
                desc.ghost_policy = static_cast<FieldGhostPolicy>(ghost_policy);
                desc.units = read_string();
                desc.description = read_string();

                const std::uint64_t n_names = read_u64();
                desc.component_names.clear();
                desc.component_names.reserve(static_cast<size_t>(n_names));
                for (std::uint64_t i = 0; i < n_names; ++i) {
                    desc.component_names.push_back(read_string());
                }
                return desc;
            };

            // Deserialize header
            const std::uint32_t magic = read_u32();
            const std::uint32_t version = read_u32();
            if (magic != kMagic) {
                throw std::runtime_error("MeshPacket::deserialize: bad magic");
            }
            if (version != kVersion) {
                throw std::runtime_error("MeshPacket::deserialize: unsupported version");
            }

	            n_cells = read_u64();
	            n_vertices = read_u64();
	            n_faces = read_u64();
	            n_edges = read_u64();
	            spatial_dim = read_i32();
	            has_current_coords = read_i32();

	            // Deserialize vertices
	            vertex_gids.resize(static_cast<size_t>(n_vertices));
	            if (n_vertices > 0) {
	                read(vertex_gids.data(), static_cast<size_t>(n_vertices) * sizeof(gid_t));
	            }

	            vertex_coords_ref.resize(static_cast<size_t>(n_vertices) * static_cast<size_t>(spatial_dim));
	            if (n_vertices > 0 && spatial_dim > 0) {
	                read(vertex_coords_ref.data(), vertex_coords_ref.size() * sizeof(real_t));
	            }

	            vertex_labels.resize(static_cast<size_t>(n_vertices));
	            if (n_vertices > 0) {
	                read(vertex_labels.data(), static_cast<size_t>(n_vertices) * sizeof(label_t));
	            }

	            if (has_current_coords != 0) {
	                vertex_coords_cur.resize(static_cast<size_t>(n_vertices) * static_cast<size_t>(spatial_dim));
	                if (n_vertices > 0 && spatial_dim > 0) {
	                    read(vertex_coords_cur.data(), vertex_coords_cur.size() * sizeof(real_t));
	                }
	            } else {
	                vertex_coords_cur.clear();
	            }

	            // Deserialize cells
            cell_gids.resize(static_cast<size_t>(n_cells));
            if (n_cells > 0) {
                read(cell_gids.data(), static_cast<size_t>(n_cells) * sizeof(gid_t));
            }

            cell_shapes.resize(static_cast<size_t>(n_cells));
            if (n_cells > 0) {
                read(cell_shapes.data(), static_cast<size_t>(n_cells) * sizeof(CellShape));
            }

            cell_offsets.resize(static_cast<size_t>(n_cells) + 1u);
            if (n_cells > 0) {
                read(cell_offsets.data(), (static_cast<size_t>(n_cells) + 1u) * sizeof(offset_t));
            } else {
                // Always expect offsets[0]==0 for empty mesh.
                cell_offsets[0] = 0;
            }

            const size_t connectivity_size = (cell_offsets.empty())
                                                ? 0u
                                                : static_cast<size_t>(cell_offsets.back());
            cell_connectivity.resize(connectivity_size);
            if (connectivity_size > 0) {
                read(cell_connectivity.data(), connectivity_size * sizeof(index_t));
            }

            cell_regions.resize(static_cast<size_t>(n_cells));
            if (n_cells > 0) {
                read(cell_regions.data(), static_cast<size_t>(n_cells) * sizeof(label_t));
            }

            cell_refinement_levels.resize(static_cast<size_t>(n_cells));
            if (n_cells > 0) {
                read(cell_refinement_levels.data(), static_cast<size_t>(n_cells) * sizeof(std::uint64_t));
            }

            // Deserialize canonical face/edge gids
            face_gids.resize(static_cast<size_t>(n_faces));
            if (n_faces > 0) {
                read(face_gids.data(), static_cast<size_t>(n_faces) * sizeof(gid_t));
            }

            edge_gids.resize(static_cast<size_t>(n_edges));
            if (n_edges > 0) {
                read(edge_gids.data(), static_cast<size_t>(n_edges) * sizeof(gid_t));
            }

            // Deserialize sparse labels
            const std::uint64_t n_face_labels = read_u64();
            face_boundary_labels.clear();
            face_boundary_labels.reserve(static_cast<size_t>(n_face_labels));
            for (std::uint64_t i = 0; i < n_face_labels; ++i) {
                gid_t gid = INVALID_GID;
                label_t label = INVALID_LABEL;
                read(&gid, sizeof(gid_t));
                read(&label, sizeof(label_t));
                face_boundary_labels.push_back({gid, label});
            }

            const std::uint64_t n_edge_labels = read_u64();
            edge_labels.clear();
            edge_labels.reserve(static_cast<size_t>(n_edge_labels));
            for (std::uint64_t i = 0; i < n_edge_labels; ++i) {
                gid_t gid = INVALID_GID;
                label_t label = INVALID_LABEL;
                read(&gid, sizeof(gid_t));
                read(&label, sizeof(label_t));
                edge_labels.push_back({gid, label});
            }

            // Deserialize sets
            const std::uint64_t n_sets = read_u64();
            sets.clear();
            sets.reserve(static_cast<size_t>(n_sets));
            for (std::uint64_t i = 0; i < n_sets; ++i) {
                SetPacket s;
                s.kind = static_cast<EntityKind>(read_i32());
                s.name = read_string();
                const std::uint64_t n_ids = read_u64();
                s.gids.resize(static_cast<size_t>(n_ids));
                if (n_ids > 0) {
                    read(s.gids.data(), static_cast<size_t>(n_ids) * sizeof(gid_t));
                }
                sets.push_back(std::move(s));
            }

            // Deserialize fields
            const std::uint64_t n_fields = read_u64();
            fields.clear();
            fields.reserve(static_cast<size_t>(n_fields));
            for (std::uint64_t i = 0; i < n_fields; ++i) {
                FieldPacket f;
                f.kind = static_cast<EntityKind>(read_i32());
                f.type = static_cast<FieldScalarType>(read_i32());
                f.components = read_u64();
                f.bytes_per_component = read_u64();
                f.name = read_string();

                const int has_desc = read_i32();
                f.has_descriptor = (has_desc != 0);
                if (f.has_descriptor) {
                    f.descriptor = read_field_descriptor();
                }

                const std::uint64_t data_size = read_u64();
                f.data.resize(static_cast<size_t>(data_size));
                if (data_size > 0) {
                    read(f.data.data(), static_cast<size_t>(data_size));
                }
                fields.push_back(std::move(f));
            }
        }
    };

	    auto build_packet_for_cells = [&](const std::vector<index_t>& cells, MeshPacket& packet) {
	        packet = MeshPacket{};
	        packet.spatial_dim = local_mesh_->dim();
	        packet.has_current_coords = local_mesh_->has_current_coords() ? 1 : 0;
	        packet.n_cells = static_cast<std::uint64_t>(cells.size());

        packet.cell_offsets.clear();
        packet.cell_offsets.push_back(0);

        if (cells.empty()) {
            packet.n_vertices = 0;
            packet.n_faces = 0;
            packet.n_edges = 0;
            return;
        }

        // Collect unique vertices used by these cells and build an old->packet vertex remap.
        std::unordered_map<index_t, index_t> vertex_remap;
        vertex_remap.reserve(cells.size() * 8);

        std::vector<index_t> vertices_for_packet;
        vertices_for_packet.reserve(cells.size() * 8);

        for (index_t c : cells) {
            auto [verts, n] = local_mesh_->cell_vertices_span(c);
            for (size_t i = 0; i < n; ++i) {
                const index_t v = verts[i];
                if (vertex_remap.find(v) == vertex_remap.end()) {
                    vertex_remap[v] = static_cast<index_t>(vertices_for_packet.size());
                    vertices_for_packet.push_back(v);
                }
            }
        }

        packet.n_vertices = static_cast<std::uint64_t>(vertices_for_packet.size());

	        // Pack vertex data
	        const auto& all_vertex_gids = local_mesh_->vertex_gids();
	        const auto& all_coords_ref = local_mesh_->X_ref();
	        const auto& all_coords_cur = local_mesh_->X_cur();

	        packet.vertex_gids.reserve(vertices_for_packet.size());
	        packet.vertex_coords_ref.reserve(vertices_for_packet.size() * static_cast<size_t>(packet.spatial_dim));
	        if (packet.has_current_coords) {
	            packet.vertex_coords_cur.reserve(vertices_for_packet.size() * static_cast<size_t>(packet.spatial_dim));
	        }
	        packet.vertex_labels.reserve(vertices_for_packet.size());

	        for (index_t v : vertices_for_packet) {
	            packet.vertex_gids.push_back(all_vertex_gids[static_cast<size_t>(v)]);
	            for (int d = 0; d < packet.spatial_dim; ++d) {
	                packet.vertex_coords_ref.push_back(all_coords_ref[static_cast<size_t>(v) * packet.spatial_dim + d]);
	                if (packet.has_current_coords) {
	                    packet.vertex_coords_cur.push_back(all_coords_cur[static_cast<size_t>(v) * packet.spatial_dim + d]);
	                }
	            }
	            packet.vertex_labels.push_back(local_mesh_->vertex_label(v));
	        }

        // Pack cell data
        const auto& all_cell_gids = local_mesh_->cell_gids();
        const auto& all_cell_shapes = local_mesh_->cell_shapes();
        const auto& all_cell_regions = local_mesh_->cell_region_ids();
        const auto& all_cell_ref_levels = local_mesh_->cell_refinement_levels();

        packet.cell_gids.reserve(cells.size());
        packet.cell_shapes.reserve(cells.size());
        packet.cell_regions.reserve(cells.size());
        packet.cell_refinement_levels.reserve(cells.size());

        for (index_t c : cells) {
            packet.cell_gids.push_back(all_cell_gids[static_cast<size_t>(c)]);
            packet.cell_shapes.push_back(all_cell_shapes[static_cast<size_t>(c)]);

            auto [verts, n] = local_mesh_->cell_vertices_span(c);
            for (size_t i = 0; i < n; ++i) {
                packet.cell_connectivity.push_back(vertex_remap[verts[i]]);
            }
            packet.cell_offsets.push_back(static_cast<offset_t>(packet.cell_connectivity.size()));

            if (!all_cell_regions.empty()) {
                packet.cell_regions.push_back(all_cell_regions[static_cast<size_t>(c)]);
            } else {
                packet.cell_regions.push_back(0);
            }

            if (all_cell_ref_levels.size() == local_mesh_->n_cells()) {
                packet.cell_refinement_levels.push_back(static_cast<std::uint64_t>(
                    all_cell_ref_levels[static_cast<size_t>(c)]));
            } else {
                packet.cell_refinement_levels.push_back(0);
            }
        }

	        // Build a temporary mesh for codim-1/codim-2 entity extraction and remapping.
	        MeshBase tmp;
	        tmp.build_from_arrays(packet.spatial_dim, packet.vertex_coords_ref, packet.cell_offsets, packet.cell_connectivity,
	                              packet.cell_shapes);
        tmp.set_vertex_gids(packet.vertex_gids);
        tmp.set_cell_gids(packet.cell_gids);
        tmp.finalize();

        ensure_canonical_face_gids(tmp);
        ensure_canonical_edge_gids(tmp);

        packet.face_gids = tmp.face_gids();
        packet.edge_gids = tmp.edge_gids();
        packet.n_faces = static_cast<std::uint64_t>(packet.face_gids.size());
        packet.n_edges = static_cast<std::uint64_t>(packet.edge_gids.size());

        // Sparse face boundary labels.
        packet.face_boundary_labels.clear();
        packet.face_boundary_labels.reserve(tmp.n_faces());
        for (size_t f = 0; f < tmp.n_faces(); ++f) {
            const gid_t gid = packet.face_gids[f];
            const index_t orig = local_mesh_->global_to_local_face(gid);
            if (orig == INVALID_INDEX) continue;
            const label_t label = local_mesh_->boundary_label(orig);
            if (label != INVALID_LABEL) {
                packet.face_boundary_labels.push_back({gid, label});
            }
        }

        // Sparse edge labels.
        packet.edge_labels.clear();
        packet.edge_labels.reserve(tmp.n_edges());
        for (size_t e = 0; e < tmp.n_edges(); ++e) {
            const gid_t gid = packet.edge_gids[e];
            const index_t orig = local_mesh_->global_to_local_edge(gid);
            if (orig == INVALID_INDEX) continue;
            const label_t label = local_mesh_->edge_label(orig);
            if (label != INVALID_LABEL) {
                packet.edge_labels.push_back({gid, label});
            }
        }

        // Sets (Vertex + Volume by entity GID).
        packet.sets.clear();
        packet.sets.reserve(16);

        for (const auto& set_name : local_mesh_->list_sets(EntityKind::Vertex)) {
            std::vector<gid_t> members;
            for (const auto v : local_mesh_->get_set(EntityKind::Vertex, set_name)) {
                const auto it = vertex_remap.find(v);
                if (it == vertex_remap.end()) continue;
                members.push_back(all_vertex_gids[static_cast<size_t>(v)]);
            }
            if (!members.empty()) {
                packet.sets.push_back({EntityKind::Vertex, set_name, std::move(members)});
            }
        }

        std::unordered_set<index_t> cell_selected;
        cell_selected.reserve(cells.size() * 2);
        for (const auto c : cells) cell_selected.insert(c);

        for (const auto& set_name : local_mesh_->list_sets(EntityKind::Volume)) {
            std::vector<gid_t> members;
            for (const auto c : local_mesh_->get_set(EntityKind::Volume, set_name)) {
                if (cell_selected.find(c) == cell_selected.end()) continue;
                members.push_back(all_cell_gids[static_cast<size_t>(c)]);
            }
            if (!members.empty()) {
                packet.sets.push_back({EntityKind::Volume, set_name, std::move(members)});
            }
        }

        // Sets (Face + Edge) by canonical GID.
        std::unordered_set<gid_t> face_set(packet.face_gids.begin(), packet.face_gids.end());
        std::unordered_set<gid_t> edge_set(packet.edge_gids.begin(), packet.edge_gids.end());

        for (const auto& set_name : local_mesh_->list_sets(EntityKind::Face)) {
            std::vector<gid_t> members;
            for (const auto f : local_mesh_->get_set(EntityKind::Face, set_name)) {
                if (f < 0 || static_cast<size_t>(f) >= local_mesh_->n_faces()) continue;
                const gid_t gid = local_mesh_->face_gids()[static_cast<size_t>(f)];
                if (face_set.find(gid) == face_set.end()) continue;
                members.push_back(gid);
            }
            if (!members.empty()) {
                packet.sets.push_back({EntityKind::Face, set_name, std::move(members)});
            }
        }

        for (const auto& set_name : local_mesh_->list_sets(EntityKind::Edge)) {
            std::vector<gid_t> members;
            for (const auto e : local_mesh_->get_set(EntityKind::Edge, set_name)) {
                if (e < 0 || static_cast<size_t>(e) >= local_mesh_->n_edges()) continue;
                const gid_t gid = local_mesh_->edge_gids()[static_cast<size_t>(e)];
                if (edge_set.find(gid) == edge_set.end()) continue;
                members.push_back(gid);
            }
            if (!members.empty()) {
                packet.sets.push_back({EntityKind::Edge, set_name, std::move(members)});
            }
        }

        // Fields (Vertex + Volume)
        packet.fields.clear();
        packet.fields.reserve(local_mesh_->field_names(EntityKind::Vertex).size() +
                              local_mesh_->field_names(EntityKind::Volume).size() +
                              local_mesh_->field_names(EntityKind::Face).size() +
                              local_mesh_->field_names(EntityKind::Edge).size());

        auto pack_direct_field = [&](EntityKind kind,
                                     const std::vector<index_t>& local_ids,
                                     const std::vector<gid_t>& gids,
                                     const std::vector<gid_t>& entity_gids_in_packet) {
            (void)gids;
            (void)entity_gids_in_packet;
            for (const auto& field_name : local_mesh_->field_names(kind)) {
                MeshPacket::FieldPacket fp;
                fp.name = field_name;
                fp.kind = kind;
                fp.type = local_mesh_->field_type_by_name(kind, field_name);

                const size_t comps = local_mesh_->field_components_by_name(kind, field_name);
                const size_t bpc = local_mesh_->field_bytes_per_component_by_name(kind, field_name);
                const size_t bpe = comps * bpc;
                fp.components = static_cast<std::uint64_t>(comps);
                fp.bytes_per_component = static_cast<std::uint64_t>(bpc);

                const auto hnd = local_mesh_->field_handle(kind, field_name);
                if (hnd.id != 0) {
                    if (const auto* desc = local_mesh_->field_descriptor(hnd)) {
                        fp.has_descriptor = true;
                        fp.descriptor = *desc;
                    }
                }

                const auto* src = static_cast<const std::uint8_t*>(local_mesh_->field_data_by_name(kind, field_name));
                if (!src) continue;

                fp.data.resize(local_ids.size() * bpe);
                for (size_t i = 0; i < local_ids.size(); ++i) {
                    const index_t id = local_ids[i];
                    std::memcpy(fp.data.data() + i * bpe,
                                src + static_cast<size_t>(id) * bpe,
                                bpe);
                }

                packet.fields.push_back(std::move(fp));
            }
        };

        pack_direct_field(EntityKind::Vertex, vertices_for_packet, packet.vertex_gids, packet.vertex_gids);
        pack_direct_field(EntityKind::Volume, cells, packet.cell_gids, packet.cell_gids);

        // Fields (Face + Edge) remapped by canonical GID.
        auto pack_gid_field = [&](EntityKind kind,
                                  const std::vector<gid_t>& gids_in_packet) {
            for (const auto& field_name : local_mesh_->field_names(kind)) {
                MeshPacket::FieldPacket fp;
                fp.name = field_name;
                fp.kind = kind;
                fp.type = local_mesh_->field_type_by_name(kind, field_name);

                const size_t comps = local_mesh_->field_components_by_name(kind, field_name);
                const size_t bpc = local_mesh_->field_bytes_per_component_by_name(kind, field_name);
                const size_t bpe = comps * bpc;
                fp.components = static_cast<std::uint64_t>(comps);
                fp.bytes_per_component = static_cast<std::uint64_t>(bpc);

                const auto hnd = local_mesh_->field_handle(kind, field_name);
                if (hnd.id != 0) {
                    if (const auto* desc = local_mesh_->field_descriptor(hnd)) {
                        fp.has_descriptor = true;
                        fp.descriptor = *desc;
                    }
                }

                const auto* src = static_cast<const std::uint8_t*>(local_mesh_->field_data_by_name(kind, field_name));
                if (!src) continue;

                fp.data.assign(gids_in_packet.size() * bpe, 0);
                for (size_t i = 0; i < gids_in_packet.size(); ++i) {
                    const gid_t gid = gids_in_packet[i];
                    const index_t orig =
                        (kind == EntityKind::Face)
                            ? local_mesh_->global_to_local_face(gid)
                            : local_mesh_->global_to_local_edge(gid);
                    if (orig == INVALID_INDEX) continue;
                    std::memcpy(fp.data.data() + i * bpe,
                                src + static_cast<size_t>(orig) * bpe,
                                bpe);
                }

                packet.fields.push_back(std::move(fp));
            }
        };

        pack_gid_field(EntityKind::Face, packet.face_gids);
        pack_gid_field(EntityKind::Edge, packet.edge_gids);
    };

    // Build local keep packet and outgoing packets.
    MeshPacket keep_packet;
    build_packet_for_cells(cells_to_keep, keep_packet);

    std::map<rank_t, MeshPacket> send_packets;
    for (const auto& [rank, cells] : cells_to_send) {
        build_packet_for_cells(cells, send_packets[rank]);
    }

    // ========================================
    // Step 4: Exchange data sizes
    // ========================================

    std::vector<int> send_sizes(world_size_, 0);
    std::vector<int> recv_sizes(world_size_, 0);

    for (const auto& [rank, packet] : send_packets) {
        std::vector<uint8_t> buffer;
        packet.serialize(buffer);
        send_sizes[rank] = static_cast<int>(buffer.size());
    }

    MPI_Alltoall(send_sizes.data(), 1, MPI_INT,
                recv_sizes.data(), 1, MPI_INT, comm_);

    // ========================================
    // Step 5: Exchange mesh data
    // ========================================

    std::vector<MPI_Request> requests;
    std::map<rank_t, std::vector<uint8_t>> send_buffers;
    std::map<rank_t, std::vector<uint8_t>> recv_buffers;

    // Serialize and send
    for (const auto& [rank, packet] : send_packets) {
        send_buffers[rank].clear();
        packet.serialize(send_buffers[rank]);

        MPI_Request req;
        MPI_Isend(send_buffers[rank].data(),
                 static_cast<int>(send_buffers[rank].size()),
                 MPI_BYTE, rank, 42, comm_, &req);
        requests.push_back(req);
    }

    // Post receives
    for (int rank = 0; rank < world_size_; ++rank) {
        if (recv_sizes[rank] > 0) {
            recv_buffers[rank].resize(recv_sizes[rank]);

            MPI_Request req;
            MPI_Irecv(recv_buffers[rank].data(),
                     recv_sizes[rank],
                     MPI_BYTE, rank, 42, comm_, &req);
            requests.push_back(req);
        }
    }

    // Wait for all communication
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    // ========================================
    // Step 6: Rebuild local mesh
    // ========================================

    // Deserialize received packets
    std::vector<MeshPacket> recv_packets;
    for (const auto& [rank, buffer] : recv_buffers) {
        if (!buffer.empty()) {
            MeshPacket packet;
            packet.deserialize(buffer);
            recv_packets.push_back(std::move(packet));
        }
    }

	    // Combine kept (local) and received data into new mesh.
	    const int spatial_dim = local_mesh_->dim();

	    bool want_current_coords = old_has_current_coords || (keep_packet.has_current_coords != 0);
	    for (const auto& packet : recv_packets) {
	        if (packet.has_current_coords != 0) {
	            want_current_coords = true;
	            break;
	        }
	    }

	    std::vector<real_t> new_coords_ref;
	    std::vector<real_t> new_coords_cur;
	    std::vector<gid_t> new_vertex_gids;
	    std::vector<CellShape> new_cell_shapes;
	    std::vector<offset_t> new_cell_offsets;
	    std::vector<index_t> new_cell_connectivity;
	    std::vector<gid_t> new_cell_gids;
	    std::vector<label_t> new_cell_regions;
	    std::vector<std::uint64_t> new_cell_refinement_levels;

	    std::unordered_map<gid_t, index_t> gid_to_new_vertex;
	    gid_to_new_vertex.reserve(keep_packet.vertex_gids.size() + 1024);
	    std::vector<std::uint8_t> current_filled;

	    auto append_vertices = [&](const MeshPacket& packet) {
	        const auto& vg = packet.vertex_gids;
	        const auto& Xref = packet.vertex_coords_ref;
	        const auto& Xcur = packet.vertex_coords_cur;
	        const int dim = packet.spatial_dim;
	        const bool packet_has_current =
	            (packet.has_current_coords != 0) &&
	            (Xcur.size() == vg.size() * static_cast<size_t>(dim));

	        for (size_t i = 0; i < vg.size(); ++i) {
	            const gid_t gid = vg[i];
	            if (gid == INVALID_GID) continue;
	            const auto it_existing = gid_to_new_vertex.find(gid);
	            if (it_existing != gid_to_new_vertex.end()) {
	                if (want_current_coords && packet_has_current) {
	                    const index_t id = it_existing->second;
	                    if (id >= 0 && static_cast<size_t>(id) < current_filled.size() &&
	                        current_filled[static_cast<size_t>(id)] == 0) {
	                        for (int d = 0; d < spatial_dim; ++d) {
	                            new_coords_cur[static_cast<size_t>(id) * spatial_dim + static_cast<size_t>(d)] =
	                                Xcur[i * static_cast<size_t>(dim) + static_cast<size_t>(d)];
	                        }
	                        current_filled[static_cast<size_t>(id)] = 1;
	                    }
	                }
	                continue;
	            }

	            const index_t new_id = static_cast<index_t>(new_vertex_gids.size());
	            gid_to_new_vertex[gid] = new_id;
	            new_vertex_gids.push_back(gid);

	            for (int d = 0; d < spatial_dim; ++d) {
	                new_coords_ref.push_back(Xref[i * static_cast<size_t>(dim) + static_cast<size_t>(d)]);
	                if (want_current_coords) {
	                    new_coords_cur.push_back(packet_has_current
	                                                 ? Xcur[i * static_cast<size_t>(dim) + static_cast<size_t>(d)]
	                                                 : Xref[i * static_cast<size_t>(dim) + static_cast<size_t>(d)]);
	                }
	            }
	            if (want_current_coords) {
	                current_filled.push_back(packet_has_current ? 1 : 0);
	            }
	        }
	    };

    append_vertices(keep_packet);
    for (const auto& packet : recv_packets) {
        append_vertices(packet);
    }

    new_cell_offsets.push_back(0);

    auto append_cells = [&](const MeshPacket& packet) {
        for (size_t c = 0; c < packet.cell_gids.size(); ++c) {
            new_cell_gids.push_back(packet.cell_gids[c]);
            new_cell_shapes.push_back(packet.cell_shapes[c]);

            const offset_t start = packet.cell_offsets[c];
            const offset_t end = packet.cell_offsets[c + 1];
            for (offset_t off = start; off < end; ++off) {
                const index_t pv = packet.cell_connectivity[static_cast<size_t>(off)];
                const gid_t gid = packet.vertex_gids[static_cast<size_t>(pv)];
                const auto it = gid_to_new_vertex.find(gid);
                if (it == gid_to_new_vertex.end()) {
                    throw std::runtime_error("migrate: missing vertex gid during connectivity rebuild");
                }
                new_cell_connectivity.push_back(it->second);
            }

            new_cell_offsets.push_back(static_cast<offset_t>(new_cell_connectivity.size()));

            if (c < packet.cell_regions.size()) {
                new_cell_regions.push_back(packet.cell_regions[c]);
            } else {
                new_cell_regions.push_back(0);
            }

            if (c < packet.cell_refinement_levels.size()) {
                new_cell_refinement_levels.push_back(packet.cell_refinement_levels[c]);
            } else {
                new_cell_refinement_levels.push_back(0);
            }
        }
    };

    append_cells(keep_packet);
    for (const auto& packet : recv_packets) {
        append_cells(packet);
    }

	    // Rebuild local mesh.
	    local_mesh_->clear();
	    local_mesh_->build_from_arrays(spatial_dim, new_coords_ref, new_cell_offsets, new_cell_connectivity, new_cell_shapes);
	    local_mesh_->set_vertex_gids(std::move(new_vertex_gids));
	    local_mesh_->set_cell_gids(std::move(new_cell_gids));

    for (size_t c = 0; c < new_cell_regions.size(); ++c) {
        local_mesh_->set_region_label(static_cast<index_t>(c), new_cell_regions[c]);
    }

    {
        std::vector<size_t> levels;
        levels.reserve(new_cell_refinement_levels.size());
        for (const auto v : new_cell_refinement_levels) {
            levels.push_back(static_cast<size_t>(v));
        }
        local_mesh_->set_cell_refinement_levels(std::move(levels));
    }

	    local_mesh_->finalize();

	    if (want_current_coords && new_coords_cur.size() == local_mesh_->n_vertices() * static_cast<size_t>(spatial_dim)) {
	        local_mesh_->set_current_coords(new_coords_cur);
	    }
	    if (old_active_config == Configuration::Current && want_current_coords) {
	        local_mesh_->use_current_configuration();
	    } else {
	        local_mesh_->use_reference_configuration();
	    }

    // Restore label registry.
    for (const auto& [label, name] : label_registry) {
        local_mesh_->register_label(name, label);
    }

    // Ensure stable codim GIDs so face/edge metadata can be restored by canonical GID.
    ensure_canonical_face_gids(*local_mesh_);
    ensure_canonical_edge_gids(*local_mesh_);

    // Restore vertex labels by GID mapping.
    auto restore_vertex_labels = [&](const MeshPacket& packet) {
        const auto& vg = packet.vertex_gids;
        const auto& vl = packet.vertex_labels;
        const size_t n = std::min(vg.size(), vl.size());
        for (size_t i = 0; i < n; ++i) {
            const label_t label = vl[i];
            if (label == INVALID_LABEL) continue;
            const index_t v = local_mesh_->global_to_local_vertex(vg[i]);
            if (v == INVALID_INDEX) continue;
            local_mesh_->set_vertex_label(v, label);
        }
    };
    restore_vertex_labels(keep_packet);
    for (const auto& packet : recv_packets) {
        restore_vertex_labels(packet);
    }

    // Restore face boundary labels and edge labels by canonical GID.
    auto restore_face_labels = [&](const MeshPacket& packet) {
        for (const auto& [gid, label] : packet.face_boundary_labels) {
            if (label == INVALID_LABEL) continue;
            const index_t f = local_mesh_->global_to_local_face(gid);
            if (f == INVALID_INDEX) continue;
            local_mesh_->set_boundary_label(f, label);
        }
    };
    auto restore_edge_labels = [&](const MeshPacket& packet) {
        for (const auto& [gid, label] : packet.edge_labels) {
            if (label == INVALID_LABEL) continue;
            const index_t e = local_mesh_->global_to_local_edge(gid);
            if (e == INVALID_INDEX) continue;
            local_mesh_->set_edge_label(e, label);
        }
    };
    restore_face_labels(keep_packet);
    restore_edge_labels(keep_packet);
    for (const auto& packet : recv_packets) {
        restore_face_labels(packet);
        restore_edge_labels(packet);
    }

    // Restore sets by mapping canonical/entity GIDs back to local indices.
    auto restore_sets = [&](const MeshPacket& packet) {
        for (const auto& set : packet.sets) {
            for (const auto gid : set.gids) {
                index_t id = INVALID_INDEX;
                switch (set.kind) {
                    case EntityKind::Vertex: id = local_mesh_->global_to_local_vertex(gid); break;
                    case EntityKind::Volume: id = local_mesh_->global_to_local_cell(gid); break;
                    case EntityKind::Face: id = local_mesh_->global_to_local_face(gid); break;
                    case EntityKind::Edge: id = local_mesh_->global_to_local_edge(gid); break;
                }
                if (id == INVALID_INDEX) continue;
                local_mesh_->add_to_set(set.kind, set.name, id);
            }
        }
    };
    restore_sets(keep_packet);
    for (const auto& packet : recv_packets) {
        restore_sets(packet);
    }

    // Restore fields (all kinds) by GID mapping. Faces/edges use canonical GIDs.
    struct FieldSource {
        const MeshPacket* packet = nullptr;
        const MeshPacket::FieldPacket* field = nullptr;
    };

    std::map<std::pair<EntityKind, std::string>, std::vector<FieldSource>> sources;

    auto index_fields = [&](const MeshPacket& packet) {
        for (const auto& field : packet.fields) {
            sources[{field.kind, field.name}].push_back({&packet, &field});
        }
    };
    index_fields(keep_packet);
    for (const auto& packet : recv_packets) {
        index_fields(packet);
    }

    for (const auto& [key, srcs] : sources) {
        const auto& kind = key.first;
        const auto& name = key.second;
        if (srcs.empty()) continue;

        const auto* meta = srcs.front().field;
        const size_t comps = static_cast<size_t>(meta->components);
        const size_t bpc = static_cast<size_t>(meta->bytes_per_component);
        const size_t bpe = comps * bpc;

        auto h = local_mesh_->attach_field(kind, name, meta->type, comps, bpc);
        auto* dst = static_cast<std::uint8_t*>(local_mesh_->field_data(h));
        if (!dst) continue;

        const size_t n_new = local_mesh_->field_entity_count(h);
        std::vector<std::uint8_t> filled(n_new, 0);

        // Apply descriptor (prefer first source that has one).
        for (const auto& src : srcs) {
            if (src.field && src.field->has_descriptor) {
                local_mesh_->set_field_descriptor(h, src.field->descriptor);
                break;
            }
        }

        auto gid_to_local = [&](gid_t gid) -> index_t {
            switch (kind) {
                case EntityKind::Vertex: return local_mesh_->global_to_local_vertex(gid);
                case EntityKind::Volume: return local_mesh_->global_to_local_cell(gid);
                case EntityKind::Face: return local_mesh_->global_to_local_face(gid);
                case EntityKind::Edge: return local_mesh_->global_to_local_edge(gid);
            }
            return INVALID_INDEX;
        };

        for (const auto& src : srcs) {
            const auto* pkt = src.packet;
            const auto* fp = src.field;
            if (!pkt || !fp) continue;
            if (bpe == 0) continue;

            const std::vector<gid_t>* gids = nullptr;
            if (kind == EntityKind::Vertex) gids = &pkt->vertex_gids;
            else if (kind == EntityKind::Volume) gids = &pkt->cell_gids;
            else if (kind == EntityKind::Face) gids = &pkt->face_gids;
            else if (kind == EntityKind::Edge) gids = &pkt->edge_gids;

            if (!gids) continue;
            const size_t n_old = gids->size();
            if (fp->data.size() < n_old * bpe) continue;

            for (size_t i = 0; i < n_old; ++i) {
                const gid_t gid = (*gids)[i];
                const index_t id = gid_to_local(gid);
                if (id == INVALID_INDEX) continue;
                if (id < 0 || static_cast<size_t>(id) >= n_new) continue;
                if (filled[static_cast<size_t>(id)]) continue;
                std::memcpy(dst + static_cast<size_t>(id) * bpe,
                            fp->data.data() + i * bpe,
                            bpe);
                filled[static_cast<size_t>(id)] = 1;
            }
        }
    }

    // Update ownership arrays
    vertex_owner_.clear();
    vertex_owner_.resize(local_mesh_->n_vertices(), Ownership::Owned);
    vertex_owner_rank_.clear();
    vertex_owner_rank_.resize(local_mesh_->n_vertices(), my_rank_);

    cell_owner_.clear();
    cell_owner_.resize(local_mesh_->n_cells(), Ownership::Owned);
    cell_owner_rank_.clear();
    cell_owner_rank_.resize(local_mesh_->n_cells(), my_rank_);

    face_owner_.clear();
    face_owner_.resize(local_mesh_->n_faces(), Ownership::Owned);
    face_owner_rank_.clear();
    face_owner_rank_.resize(local_mesh_->n_faces(), my_rank_);

    edge_owner_.clear();
    edge_owner_.resize(local_mesh_->n_edges(), Ownership::Owned);
    edge_owner_rank_.clear();
    edge_owner_rank_.resize(local_mesh_->n_edges(), my_rank_);

    // ========================================
	    // Step 7: Rebuild ghost layers
	    // ========================================

	    gather_shared_entities();
	    build_ghost_layer(ghost_levels_);

	    if (local_mesh_) {
	        local_mesh_->event_bus().notify(MeshEvent::PartitionChanged);
	    }
	#endif
	}

void DistributedMesh::rebalance(PartitionHint hint,
                               const std::unordered_map<std::string,std::string>& options) {
#ifdef MESH_HAS_MPI
    if (world_size_ == 1) {
        return;
    }
    if (!local_mesh_) {
        return;
    }

    // Rebalancing must operate on the set of *globally owned* cells only. Including ghosts
    // contaminates global counts and can lead to duplicated migration decisions.
    if (cell_owner_.size() != local_mesh_->n_cells()) {
        cell_owner_.assign(local_mesh_->n_cells(), Ownership::Owned);
    }
    if (cell_owner_rank_.size() != local_mesh_->n_cells()) {
        cell_owner_rank_.assign(local_mesh_->n_cells(), my_rank_);
    }

    auto is_locally_owned_cell = [&](index_t c) -> bool {
        if (c < 0) return false;
        if (static_cast<size_t>(c) >= local_mesh_->n_cells()) return false;
        if (cell_owner_.size() == local_mesh_->n_cells() &&
            cell_owner_[static_cast<size_t>(c)] == Ownership::Ghost) {
            return false;
        }
        if (cell_owner_rank_.size() == local_mesh_->n_cells()) {
            return cell_owner_rank_[static_cast<size_t>(c)] == my_rank_;
        }
        return true;
    };

    // Collect locally owned cell IDs (by local index) + weights for global coordination.
    std::vector<index_t> owned_cells;
    std::vector<gid_t> owned_gids;
    std::vector<std::uint64_t> owned_weights;

    owned_cells.reserve(local_mesh_->n_cells());
    owned_gids.reserve(local_mesh_->n_cells());
    owned_weights.reserve(local_mesh_->n_cells());

    PartitionHint weight_hint = hint;
    if (weight_hint == PartitionHint::Metis || weight_hint == PartitionHint::ParMetis) {
        weight_hint = PartitionHint::Cells;
    }
    if (const auto it = options.find("partition_weight"); it != options.end()) {
        std::string w = it->second;
        std::transform(w.begin(), w.end(), w.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (w == "cells" || w == "cell") {
            weight_hint = PartitionHint::Cells;
        } else if (w == "vertices" || w == "vertex") {
            weight_hint = PartitionHint::Vertices;
        } else if (w == "memory" || w == "mem") {
            weight_hint = PartitionHint::Memory;
        }
    }

    // Optional user-provided per-cell weight field (scalar attached to Volume entities).
    // This overrides the heuristic weights derived from `weight_hint`.
    std::string cell_weight_field;
    FieldHandle cell_weight_handle;
    FieldScalarType cell_weight_type = FieldScalarType::Custom;
    const void* cell_weight_data = nullptr;

    if (const auto it = options.find("cell_weight_field"); it != options.end()) {
        cell_weight_field = it->second;
    }
    if (!cell_weight_field.empty()) {
        cell_weight_handle = local_mesh_->field_handle(EntityKind::Volume, cell_weight_field);
        if (cell_weight_handle.id == 0) {
            throw std::runtime_error("DistributedMesh::rebalance: cell_weight_field not found: " + cell_weight_field);
        }
        if (local_mesh_->field_components(cell_weight_handle) != 1) {
            throw std::runtime_error("DistributedMesh::rebalance: cell_weight_field must be scalar");
        }
        cell_weight_type = local_mesh_->field_type(cell_weight_handle);
        cell_weight_data = local_mesh_->field_data(cell_weight_handle);
        if (!cell_weight_data) {
            throw std::runtime_error("DistributedMesh::rebalance: cell_weight_field has no data");
        }
    }
    const bool has_custom_cell_weights = (cell_weight_handle.id != 0);

    auto weight_for_cell = [&](index_t c) -> std::uint64_t {
        if (has_custom_cell_weights) {
            if (c < 0 || static_cast<size_t>(c) >= local_mesh_->n_cells()) {
                return 1;
            }

            const size_t idx = static_cast<size_t>(c);
            switch (cell_weight_type) {
                case FieldScalarType::Float64: {
                    const auto* w = static_cast<const double*>(cell_weight_data);
                    const auto v = static_cast<std::int64_t>(std::llround(w[idx]));
                    return static_cast<std::uint64_t>(std::max<std::int64_t>(1, v));
                }
                case FieldScalarType::Float32: {
                    const auto* w = static_cast<const float*>(cell_weight_data);
                    const auto v = static_cast<std::int64_t>(std::llround(static_cast<double>(w[idx])));
                    return static_cast<std::uint64_t>(std::max<std::int64_t>(1, v));
                }
                case FieldScalarType::Int32: {
                    const auto* w = static_cast<const std::int32_t*>(cell_weight_data);
                    return static_cast<std::uint64_t>(std::max<std::int32_t>(1, w[idx]));
                }
                case FieldScalarType::Int64: {
                    const auto* w = static_cast<const std::int64_t*>(cell_weight_data);
                    return static_cast<std::uint64_t>(std::max<std::int64_t>(1, w[idx]));
                }
                case FieldScalarType::UInt8: {
                    const auto* w = static_cast<const std::uint8_t*>(cell_weight_data);
                    return static_cast<std::uint64_t>(std::max<std::uint8_t>(1, w[idx]));
                }
                case FieldScalarType::Custom:
                    break;
            }
        }

        if (weight_hint == PartitionHint::Cells) {
            return 1;
        }
        if (weight_hint == PartitionHint::Vertices) {
            auto [vptr, n] = local_mesh_->cell_vertices_span(c);
            (void)vptr;
            return std::max<std::uint64_t>(1, static_cast<std::uint64_t>(n));
        }
        if (weight_hint == PartitionHint::Memory) {
            // Memory footprint approximation for one cell (mirrors the legacy heuristic but is
            // coordinated globally to avoid per-rank independent greedy assignments).
            std::uint64_t mem = 0;
            auto [vptr, n] = local_mesh_->cell_vertices_span(c);
            (void)vptr;
            mem += sizeof(CellShape);
            mem += static_cast<std::uint64_t>(n) * sizeof(index_t);
            mem += sizeof(gid_t);
            mem += sizeof(label_t);
            mem += static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(local_mesh_->dim()) *
                   sizeof(real_t) / 4u;

            for (const auto& field_name : local_mesh_->field_names(EntityKind::Volume)) {
                const std::uint64_t comps =
                    static_cast<std::uint64_t>(local_mesh_->field_components_by_name(EntityKind::Volume, field_name));
                const std::uint64_t bpc =
                    static_cast<std::uint64_t>(local_mesh_->field_bytes_per_component_by_name(EntityKind::Volume, field_name));
                mem += comps * bpc;
            }

            // Approximate overhead for halo/ghost metadata.
            mem = static_cast<std::uint64_t>(static_cast<long double>(mem) * 1.2L);
            return std::max<std::uint64_t>(1, mem);
        }

        // Graph-based partitioning is not available without a true parallel partitioner.
        // Fall back to cell-count balancing (stable and globally coordinated).
        return 1;
    };

    const auto& cell_gids = local_mesh_->cell_gids();
    for (index_t c = 0; c < static_cast<index_t>(local_mesh_->n_cells()); ++c) {
        if (!is_locally_owned_cell(c)) continue;
        owned_cells.push_back(c);
        owned_gids.push_back(cell_gids[static_cast<size_t>(c)]);
        owned_weights.push_back(weight_for_cell(c));
    }

    const int local_owned = static_cast<int>(owned_cells.size());

    const bool wants_graph_partition =
        (hint == PartitionHint::Metis || hint == PartitionHint::ParMetis);

    std::string partition_method;
    if (const auto it = options.find("partition_method"); it != options.end()) {
        partition_method = it->second;
        std::transform(partition_method.begin(), partition_method.end(), partition_method.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    }

    const bool wants_parmetis =
        wants_graph_partition || (partition_method == "parmetis");

#if defined(SVMP_HAS_PARMETIS)
    // =====================================================
    // True parallel graph partitioning via ParMETIS
    // =====================================================
    if (wants_parmetis) {
        const int saved_ghost_levels = ghost_levels_;

        // Algorithm selection:
        // - mesh : ParMETIS_V3_PartMeshKway (legacy-like, builds dual internally from cell->vertex)
        // - graph: ParMETIS_V3_PartKway on an explicit dual graph (requires ghost layer)
        std::string parmetis_algorithm = "mesh";
        if (const auto it = options.find("parmetis_algorithm"); it != options.end()) {
            parmetis_algorithm = it->second;
            std::transform(parmetis_algorithm.begin(), parmetis_algorithm.end(),
                           parmetis_algorithm.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        }

        const bool want_mesh2dual =
            (parmetis_algorithm.empty() ||
             parmetis_algorithm == "mesh" ||
             parmetis_algorithm == "mesh2dual" ||
             parmetis_algorithm == "partmesh" ||
             parmetis_algorithm == "partmeshkway");

        // Prefer the legacy-like mesh partitioner when requested; it avoids building an explicit
        // ghost layer and gives ParMETIS the native cell->vertex incidence.
        if (want_mesh2dual) {
            // Ensure ownership and exchange patterns are valid before we assign global node IDs.
            if (vertex_exchange_.send_ranks.empty() && vertex_exchange_.recv_ranks.empty() &&
                cell_exchange_.send_ranks.empty() && cell_exchange_.recv_ranks.empty() &&
                face_exchange_.send_ranks.empty() && face_exchange_.recv_ranks.empty() &&
                edge_exchange_.send_ranks.empty() && edge_exchange_.recv_ranks.empty()) {
                build_exchange_patterns();
            }

            // Determine which ranks actually have owned elements. ParMETIS mesh routines require
            // at least one element per participating rank (legacy solver excludes empty ranks).
            const std::uint64_t local_elems = static_cast<std::uint64_t>(owned_cells.size());
            std::vector<std::uint64_t> elem_counts(static_cast<size_t>(world_size_), 0);
            MPI_Allgather(&local_elems, 1, MPI_UINT64_T,
                          elem_counts.data(), 1, MPI_UINT64_T, comm_);

            std::vector<int> active_ranks;
            active_ranks.reserve(static_cast<size_t>(world_size_));
            for (int r = 0; r < world_size_; ++r) {
                if (elem_counts[static_cast<size_t>(r)] > 0) {
                    active_ranks.push_back(r);
                }
            }

            const int active_size = static_cast<int>(active_ranks.size());
            const bool is_active = (local_elems > 0);

            MPI_Comm pm_comm = comm_;
            MPI_Comm active_comm = MPI_COMM_NULL;
            std::vector<int> active_index(static_cast<size_t>(world_size_), -1);
            for (int i = 0; i < active_size; ++i) {
                active_index[static_cast<size_t>(active_ranks[static_cast<size_t>(i)])] = i;
            }

            if (active_size >= 2 && active_size < world_size_) {
                MPI_Group world_group;
                MPI_Comm_group(comm_, &world_group);
                MPI_Group incl_group;
                MPI_Group_incl(world_group, active_size, active_ranks.data(), &incl_group);
                MPI_Comm_create(comm_, incl_group, &active_comm);
                MPI_Group_free(&incl_group);
                MPI_Group_free(&world_group);
                if (active_comm != MPI_COMM_NULL) {
                    pm_comm = active_comm;
                }
                if (options.find("verbose") != options.end() && my_rank_ == 0) {
                    std::cerr << "Note: ParMETIS mesh partitioning excluded " << (world_size_ - active_size)
                              << " empty ranks.\n";
                }
            }

            int mesh_status = METIS_ERROR;
            std::vector<rank_t> new_owner_rank_per_cell(local_mesh_->n_cells(), my_rank_);

            // Compute contiguous global node IDs for vertices (owner assigns, then exchange).
            // This must be done collectively over the full communicator because the ownership
            // rules and exchange patterns are defined on `comm_` (even when we partition on an
            // active subcommunicator).
            std::vector<::idx_t> global_node_id_per_vertex(local_mesh_->n_vertices(), static_cast<::idx_t>(-1));
            if (active_size >= 2) {
                std::uint64_t local_owned_vertices = 0;
                for (index_t v = 0; v < static_cast<index_t>(local_mesh_->n_vertices()); ++v) {
                    if (is_owned_vertex(v)) {
                        ++local_owned_vertices;
                    }
                }

                std::vector<std::uint64_t> vertex_counts(static_cast<size_t>(world_size_), 0);
                MPI_Allgather(&local_owned_vertices, 1, MPI_UINT64_T,
                              vertex_counts.data(), 1, MPI_UINT64_T, comm_);

                std::uint64_t vertex_prefix = 0;
                for (int r = 0; r < my_rank_; ++r) {
                    vertex_prefix += vertex_counts[static_cast<size_t>(r)];
                }

                ::idx_t next = checked_idx_cast(vertex_prefix, "ParMETIS node offset");
                for (index_t v = 0; v < static_cast<index_t>(local_mesh_->n_vertices()); ++v) {
                    if (!is_owned_vertex(v)) continue;
                    global_node_id_per_vertex[static_cast<size_t>(v)] = next;
                    ++next;
                }

                exchange_entity_data(EntityKind::Vertex,
                                     global_node_id_per_vertex.data(),
                                     global_node_id_per_vertex.data(),
                                     sizeof(::idx_t),
                                     vertex_exchange_);
            }

            if (active_size >= 2 && is_active && pm_comm != MPI_COMM_NULL) {
                // Build elmdist for the participating ranks (active communicator ordering).
                std::vector<::idx_t> elmdist(static_cast<size_t>(active_size) + 1, 0);
                std::uint64_t prefix = 0;
                elmdist[0] = 0;
                for (int i = 0; i < active_size; ++i) {
                    prefix += elem_counts[static_cast<size_t>(active_ranks[static_cast<size_t>(i)])];
                    elmdist[static_cast<size_t>(i) + 1] = checked_idx_cast(prefix, "ParMETIS elmdist");
                }

                // Build eptr/eind for owned elements using the contiguous node IDs.
                const size_t n_owned = owned_cells.size();
                std::vector<::idx_t> eptr(n_owned + 1, 0);
                std::vector<::idx_t> eind;
                eind.reserve(n_owned * 8);

                std::uint64_t cursor = 0;
                for (size_t i = 0; i < n_owned; ++i) {
                    const index_t c = owned_cells[i];
                    auto [vptr, nv] = local_mesh_->cell_vertices_span(c);
                    for (size_t k = 0; k < nv; ++k) {
                        const index_t v = vptr[k];
                        if (v < 0 || static_cast<size_t>(v) >= global_node_id_per_vertex.size()) {
                            throw std::runtime_error("DistributedMesh::rebalance: invalid vertex index in cell");
                        }
                        const ::idx_t gid = global_node_id_per_vertex[static_cast<size_t>(v)];
                        if (gid < 0) {
                            throw std::runtime_error("DistributedMesh::rebalance: missing global node ID for vertex");
                        }
                        eind.push_back(gid);
                    }
                    cursor += nv;
                    eptr[i + 1] = checked_idx_cast(cursor, "ParMETIS eptr");
                }

                // Compute ncommonnodes: default to the minimum face-node count across the mesh (global).
                ::idx_t ncommonnodes = 0;
                if (const auto it = options.find("parmetis_ncommonnodes"); it != options.end()) {
                    ncommonnodes = static_cast<::idx_t>(std::stoi(it->second));
                } else {
                    int local_min = std::numeric_limits<int>::max();
                    for (size_t i = 0; i < n_owned; ++i) {
                        const index_t c = owned_cells[i];
                        const auto& cs = local_mesh_->cell_shape(c);
                        auto [vptr, nv] = local_mesh_->cell_vertices_span(c);
                        (void)vptr;

                        int p = std::max(1, cs.order);
                        CellTopology::HighOrderKind kind = CellTopology::HighOrderKind::Lagrange;
                        if (nv > static_cast<size_t>(cs.num_corners)) {
                            const int p_ser = CellTopology::infer_serendipity_order(cs.family, nv);
                            const int p_lag = CellTopology::infer_lagrange_order(cs.family, nv);
                            if (p_ser > 0) {
                                p = p_ser;
                                kind = CellTopology::HighOrderKind::Serendipity;
                            } else if (p_lag > 0) {
                                p = p_lag;
                            }
                        }

                        int cell_min = std::numeric_limits<int>::max();
                        try {
                            const auto faces = CellTopology::get_oriented_boundary_faces(cs.family);
                            for (int f = 0; f < static_cast<int>(faces.size()); ++f) {
                                int n = 0;
                                if (p <= 1) {
                                    n = static_cast<int>(faces[static_cast<size_t>(f)].size());
                                } else {
                                    n = static_cast<int>(CellTopology::high_order_face_local_nodes(cs.family, p, f, kind).size());
                                }
                                cell_min = std::min(cell_min, n);
                            }
                        } catch (...) {
                            cell_min = std::max(1, local_mesh_->dim());
                        }
                        if (cell_min == std::numeric_limits<int>::max()) {
                            cell_min = std::max(1, local_mesh_->dim());
                        }
                        local_min = std::min(local_min, cell_min);
                    }

                    if (local_min == std::numeric_limits<int>::max()) {
                        local_min = std::max(1, local_mesh_->dim());
                    }

                    int global_min = local_min;
                    MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, pm_comm);
                    ncommonnodes = checked_idx_cast(static_cast<std::uint64_t>(std::max(1, global_min)),
                                                   "ParMETIS ncommonnodes");
                }

                // Vertex weights (optional).
                std::vector<::idx_t> elmwgt;
                ::idx_t wgtflag = 0;
                ::idx_t* elmwgt_ptr = nullptr;
                if (has_custom_cell_weights || weight_hint == PartitionHint::Vertices || weight_hint == PartitionHint::Memory) {
                    wgtflag = 2;
                    elmwgt.resize(n_owned, 1);
                    for (size_t i = 0; i < n_owned; ++i) {
                        const std::uint64_t w = weight_for_cell(owned_cells[i]);
                        elmwgt[i] = checked_idx_cast(std::max<std::uint64_t>(1, w), "ParMETIS elmwgt");
                    }
                    elmwgt_ptr = elmwgt.data();
                }

                ::idx_t numflag = 0;
                ::idx_t ncon = 1;
                ::idx_t nparts = checked_idx_cast(static_cast<std::uint64_t>(active_size), "ParMETIS nparts");

                std::vector<::real_t> tpwgts(static_cast<size_t>(ncon) * static_cast<size_t>(nparts),
                                             static_cast<::real_t>(1.0) / static_cast<::real_t>(nparts));
                std::vector<::real_t> ubvec(static_cast<size_t>(ncon), static_cast<::real_t>(1.05));

                std::array<::idx_t, 4> pm_options{{0, 0, 0, 0}};
                pm_options[0] = 1;
                pm_options[1] = 0;
                pm_options[2] = 42;
                pm_options[3] = 0;

                std::vector<::idx_t> part(n_owned, checked_idx_cast(static_cast<std::uint64_t>(active_index[static_cast<size_t>(my_rank_)]),
                                                                   "ParMETIS initial part"));
                ::idx_t edgecut = 0;
                MPI_Comm comm = pm_comm;

                mesh_status =
                    ParMETIS_V3_PartMeshKway(elmdist.data(),
                                             eptr.data(),
                                             eind.data(),
                                             elmwgt_ptr,
                                             &wgtflag,
                                             &numflag,
                                             &ncon,
                                             &ncommonnodes,
                                             &nparts,
                                             tpwgts.data(),
                                             ubvec.data(),
                                             pm_options.data(),
                                             &edgecut,
                                             part.data(),
                                             &comm);

                if (mesh_status == METIS_OK) {
                    for (size_t i = 0; i < n_owned; ++i) {
                        const auto p = part[i];
                        if (p < 0 || p >= active_size) continue;
                        new_owner_rank_per_cell[static_cast<size_t>(owned_cells[i])] =
                            static_cast<rank_t>(active_ranks[static_cast<size_t>(p)]);
                    }
                }
            }

            int local_ok = 1;
            if (active_size >= 2 && is_active) {
                local_ok = (mesh_status == METIS_OK) ? 1 : 0;
            }
            int global_ok = 0;
            MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm_);

            if (active_comm != MPI_COMM_NULL) {
                MPI_Comm_free(&active_comm);
            }

            if (active_size >= 2 && global_ok == 1) {
                ghost_levels_ = saved_ghost_levels;
                migrate(new_owner_rank_per_cell);
                if (local_mesh_) {
                    local_mesh_->event_bus().notify(MeshEvent::PartitionChanged);
                }
                return;
            }

            if (active_size >= 2 && global_ok == 0 && options.find("verbose") != options.end() && my_rank_ == 0) {
                std::cerr << "Warning: ParMETIS_V3_PartMeshKway failed; falling back to graph/heuristic partitioning.\n";
            }
        }

        // Graph-based ParMETIS needs cross-rank adjacency. Ensure at least a 1-deep ghost layer so
        // partition interfaces are visible in the local dual graph.
        if (ghost_levels_ == 0) {
            build_ghost_layer(1);
        }

        // Recompute owned cells after any ghost-layer modifications.
        std::vector<index_t> owned_cells_graph;
        owned_cells_graph.reserve(local_mesh_->n_cells());
        for (index_t c = 0; c < static_cast<index_t>(local_mesh_->n_cells()); ++c) {
            if (!is_locally_owned_cell(c)) continue;
            owned_cells_graph.push_back(c);
        }

        const std::uint64_t local_n = static_cast<std::uint64_t>(owned_cells_graph.size());
        std::vector<std::uint64_t> counts(static_cast<size_t>(world_size_), 0);
        MPI_Allgather(&local_n, 1, MPI_UINT64_T,
                      counts.data(), 1, MPI_UINT64_T, comm_);

        std::uint64_t global_n_u64 = 0;
        for (const auto n : counts) global_n_u64 += n;

        // Degenerate case: not enough cells to split; fall back to coordinated balance.
        if (global_n_u64 == 0 || global_n_u64 < static_cast<std::uint64_t>(world_size_)) {
            ghost_levels_ = saved_ghost_levels;
        } else {
            std::vector<::idx_t> vtxdist(static_cast<size_t>(world_size_) + 1, 0);
            std::uint64_t prefix = 0;
            vtxdist[0] = 0;
            for (int r = 0; r < world_size_; ++r) {
                prefix += counts[static_cast<size_t>(r)];
                vtxdist[static_cast<size_t>(r) + 1] = checked_idx_cast(prefix, "ParMETIS vtxdist");
            }

            const ::idx_t my_offset = vtxdist[static_cast<size_t>(my_rank_)];
            std::vector<::idx_t> global_graph_id_per_cell(local_mesh_->n_cells(), static_cast<::idx_t>(-1));

            for (size_t i = 0; i < owned_cells_graph.size(); ++i) {
                const index_t c = owned_cells_graph[i];
                global_graph_id_per_cell[static_cast<size_t>(c)] =
                    my_offset + checked_idx_cast(static_cast<std::uint64_t>(i), "ParMETIS local vertex index");
            }

            // Populate global graph IDs on ghost cells so local adjacency can reference
            // neighbor vertices via their owner's numbering in vtxdist.
            exchange_entity_data(EntityKind::Volume,
                                 global_graph_id_per_cell.data(),
                                 global_graph_id_per_cell.data(),
                                 sizeof(::idx_t),
                                 cell_exchange_);

            // Build local dual graph adjacency using the existing topology builder.
            std::vector<offset_t> cell2cell_offsets;
            std::vector<index_t> cell2cell;
            MeshTopology::build_cell2cell(*local_mesh_, cell2cell_offsets, cell2cell);

            const size_t n_owned = owned_cells_graph.size();
            std::vector<::idx_t> xadj(n_owned + 1, 0);
            std::vector<::idx_t> adjncy;
            adjncy.reserve(n_owned * 8);

            std::uint64_t cursor = 0;
            for (size_t i = 0; i < n_owned; ++i) {
                const index_t c = owned_cells_graph[i];
                const auto start = static_cast<size_t>(cell2cell_offsets[static_cast<size_t>(c)]);
                const auto end = static_cast<size_t>(cell2cell_offsets[static_cast<size_t>(c + 1)]);

                std::vector<::idx_t> row;
                row.reserve(end - start);

                const ::idx_t my_gid = global_graph_id_per_cell[static_cast<size_t>(c)];
                for (size_t k = start; k < end; ++k) {
                    const index_t nbr = cell2cell[k];
                    if (nbr == c) continue;
                    if (nbr < 0 || static_cast<size_t>(nbr) >= local_mesh_->n_cells()) continue;
                    const ::idx_t ngid = global_graph_id_per_cell[static_cast<size_t>(nbr)];
                    if (ngid < 0) continue;
                    if (ngid == my_gid) continue;
                    row.push_back(ngid);
                }

                std::sort(row.begin(), row.end());
                row.erase(std::unique(row.begin(), row.end()), row.end());

                adjncy.insert(adjncy.end(), row.begin(), row.end());
                cursor += row.size();
                xadj[i + 1] = checked_idx_cast(cursor, "ParMETIS xadj");
            }

            std::uint64_t local_edges = static_cast<std::uint64_t>(adjncy.size());
            std::uint64_t global_edges = 0;
            MPI_Allreduce(&local_edges, &global_edges, 1, MPI_UINT64_T, MPI_SUM, comm_);

            // If the dual graph has no edges, ParMETIS will produce essentially arbitrary
            // parts; prefer the coordinated load-balance path below.
            if (global_edges == 0) {
                ghost_levels_ = saved_ghost_levels;
            } else {
                ::idx_t wgtflag = 0;
                ::idx_t numflag = 0;
                ::idx_t ncon = 1;
                ::idx_t nparts = checked_idx_cast(static_cast<std::uint64_t>(world_size_), "ParMETIS nparts");

                std::vector<::idx_t> vwgt;
                ::idx_t* vwgt_ptr = nullptr;
                if (has_custom_cell_weights || weight_hint == PartitionHint::Vertices || weight_hint == PartitionHint::Memory) {
                    wgtflag = 2;
                    vwgt.resize(n_owned, 1);
                    for (size_t i = 0; i < n_owned; ++i) {
                        const std::uint64_t w = weight_for_cell(owned_cells_graph[i]);
                        vwgt[i] = checked_idx_cast(std::max<std::uint64_t>(1, w), "ParMETIS vwgt");
                    }
                    vwgt_ptr = vwgt.data();
                }

                std::vector<::real_t> tpwgts(static_cast<size_t>(ncon) * static_cast<size_t>(nparts),
                                             static_cast<::real_t>(1.0) / static_cast<::real_t>(nparts));
                std::vector<::real_t> ubvec(static_cast<size_t>(ncon), static_cast<::real_t>(1.05));

                std::array<::idx_t, 4> pm_options{{0, 0, 0, 0}};
                pm_options[0] = 1;
                // ParMETIS v3 options array:
                // [0]=use supplied options, [1]=dbglvl, [2]=seed, [3]=ipart/psr.
                pm_options[1] = 0;
                pm_options[2] = 42;
                pm_options[3] = 0;

                std::vector<::idx_t> part(n_owned, static_cast<::idx_t>(my_rank_));
                ::idx_t edgecut = 0;
                MPI_Comm comm = comm_;

                const int status =
                    ParMETIS_V3_PartKway(vtxdist.data(),
                                         xadj.data(),
                                         adjncy.data(),
                                         /*vwgt=*/vwgt_ptr,
                                         /*adjwgt=*/nullptr,
                                         &wgtflag,
                                         &numflag,
                                         &ncon,
                                         &nparts,
                                         tpwgts.data(),
                                         ubvec.data(),
                                         pm_options.data(),
                                         &edgecut,
                                         part.data(),
                                         &comm);

                if (status == METIS_OK) {
                    std::vector<rank_t> new_owner_rank_per_cell(local_mesh_->n_cells(), my_rank_);
                    for (size_t i = 0; i < n_owned; ++i) {
                        const auto target = part[i];
                        if (target < 0 || target >= world_size_) {
                            continue;
                        }
                        new_owner_rank_per_cell[static_cast<size_t>(owned_cells_graph[i])] =
                            static_cast<rank_t>(target);
                    }

                    // Restore the caller's ghost-layer policy before migrating so migrate()
                    // rebuilds to the intended depth.
                    ghost_levels_ = saved_ghost_levels;
                    migrate(new_owner_rank_per_cell);
                    if (local_mesh_) {
                        local_mesh_->event_bus().notify(MeshEvent::PartitionChanged);
                    }
                    return;
                }

                if (options.find("verbose") != options.end()) {
                    std::cerr << "Warning: ParMETIS_V3_PartKway failed; falling back to coordinated load balancing.\n";
                }

                ghost_levels_ = saved_ghost_levels;
            }
        }
    }
#else
    if (wants_parmetis && options.find("verbose") != options.end()) {
        std::cerr << "Warning: ParMETIS partitioning requested but ParMETIS is not enabled; "
                     "falling back to coordinated load balancing.\n";
    }
#endif

    // Fast path (no gathers): when cell GIDs form a global [0..N-1] numbering, we can
    // compute a deterministic block partition in O(n_local) time.
    //
    // This keeps the common "contiguous global IDs" case comparable to legacy performance,
    // while still providing a correct fallback for arbitrary GID ranges below.
    {
        const std::uint64_t local_n = static_cast<std::uint64_t>(owned_cells.size());
        std::uint64_t global_n = 0;
        MPI_Allreduce(&local_n, &global_n, 1, MPI_UINT64_T, MPI_SUM, comm_);

        if (global_n > 0 && hint == PartitionHint::Cells) {
            gid_t local_min = std::numeric_limits<gid_t>::max();
            gid_t local_max = std::numeric_limits<gid_t>::min();
            bool local_in_range = true;
            for (const auto gid : owned_gids) {
                local_min = std::min(local_min, gid);
                local_max = std::max(local_max, gid);
                if (gid < 0 || static_cast<std::uint64_t>(gid) >= global_n) {
                    local_in_range = false;
                }
            }
            gid_t global_min = 0;
            gid_t global_max = 0;
            MPI_Allreduce(&local_min, &global_min, 1, MPI_INT64_T, MPI_MIN, comm_);
            MPI_Allreduce(&local_max, &global_max, 1, MPI_INT64_T, MPI_MAX, comm_);

            int local_ok = local_in_range ? 1 : 0;
            int global_ok = 0;
            MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, comm_);

            const bool looks_contiguous =
                (global_ok != 0) &&
                (global_min == 0) &&
                (global_max == static_cast<gid_t>(global_n - 1));

            if (looks_contiguous) {
                std::vector<rank_t> new_owner_rank_per_cell(local_mesh_->n_cells(), my_rank_);
                const std::uint64_t cells_per_rank = global_n / static_cast<std::uint64_t>(world_size_);
                const std::uint64_t extra = global_n % static_cast<std::uint64_t>(world_size_);

                auto rank_for_global_id = [&](std::uint64_t gid) -> rank_t {
                    const std::uint64_t first_block = (cells_per_rank + 1u) * extra;
                    if (gid < first_block) {
                        return static_cast<rank_t>(gid / (cells_per_rank + 1u));
                    }
                    // cells_per_rank can be 0 only when gid < first_block (handled above).
                    return static_cast<rank_t>(extra + (gid - first_block) / cells_per_rank);
                };

                for (size_t i = 0; i < owned_cells.size(); ++i) {
                    const gid_t gid = owned_gids[i];
                    if (gid < 0) continue;
                    new_owner_rank_per_cell[static_cast<size_t>(owned_cells[i])] =
                        rank_for_global_id(static_cast<std::uint64_t>(gid));
                }

                migrate(new_owner_rank_per_cell);
                if (local_mesh_) {
                    local_mesh_->event_bus().notify(MeshEvent::PartitionChanged);
                }
                return;
            }
        }
    }

    // Gather owned cell (gid, weight) lists on rank 0 to compute a globally consistent
    // assignment. This avoids gid-contiguity assumptions and avoids per-rank independent
    // heuristics that can overload a single destination rank.
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    if (my_rank_ == 0) {
        recv_counts.assign(world_size_, 0);
    }

    MPI_Gather(&local_owned, 1, MPI_INT,
               my_rank_ == 0 ? recv_counts.data() : nullptr, 1, MPI_INT,
               0, comm_);

    int total_owned = 0;
    if (my_rank_ == 0) {
        recv_displs.assign(world_size_ + 1, 0);
        for (int r = 0; r < world_size_; ++r) {
            recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
        }
        total_owned = recv_displs[world_size_];
    }

    std::vector<gid_t> all_gids;
    std::vector<std::uint64_t> all_weights;
    if (my_rank_ == 0) {
        all_gids.assign(static_cast<size_t>(total_owned), INVALID_GID);
        all_weights.assign(static_cast<size_t>(total_owned), 1);
    }

    MPI_Gatherv(owned_gids.data(), local_owned, MPI_INT64_T,
                my_rank_ == 0 ? all_gids.data() : nullptr,
                my_rank_ == 0 ? recv_counts.data() : nullptr,
                my_rank_ == 0 ? recv_displs.data() : nullptr,
                MPI_INT64_T, 0, comm_);

    MPI_Gatherv(owned_weights.data(), local_owned, MPI_UINT64_T,
                my_rank_ == 0 ? all_weights.data() : nullptr,
                my_rank_ == 0 ? recv_counts.data() : nullptr,
                my_rank_ == 0 ? recv_displs.data() : nullptr,
                MPI_UINT64_T, 0, comm_);

    // Compute assignment on rank 0 and scatter assignments back to the current owners.
    std::vector<int> assigned_flat;
    if (my_rank_ == 0) {
        assigned_flat.assign(static_cast<size_t>(total_owned), 0);

        struct CellItem {
            gid_t gid = INVALID_GID;
            std::uint64_t weight = 1;
            int src_rank = 0;
            int src_pos = 0;
        };

        std::vector<CellItem> items;
        items.reserve(static_cast<size_t>(total_owned));

        for (int r = 0; r < world_size_; ++r) {
            const int n = recv_counts[r];
            for (int i = 0; i < n; ++i) {
                const int idx = recv_displs[r] + i;
                items.push_back({all_gids[static_cast<size_t>(idx)],
                                 std::max<std::uint64_t>(1, all_weights[static_cast<size_t>(idx)]),
                                 r,
                                 i});
            }
        }

        std::sort(items.begin(), items.end(),
                  [](const CellItem& a, const CellItem& b) { return a.gid < b.gid; });

        std::uint64_t total_weight = 0;
        for (const auto& it : items) total_weight += it.weight;
        if (total_weight == 0) total_weight = static_cast<std::uint64_t>(items.size());

        const std::uint64_t base = total_weight / static_cast<std::uint64_t>(world_size_);
        const std::uint64_t extra = total_weight % static_cast<std::uint64_t>(world_size_);

        rank_t cur_rank = 0;
        std::uint64_t cur_weight = 0;
        auto target_for = [&](rank_t r) -> std::uint64_t {
            return base + (static_cast<std::uint64_t>(r) < extra ? 1u : 0u);
        };
        std::uint64_t cur_target = target_for(cur_rank);

        for (const auto& it : items) {
            const std::uint64_t w = it.weight;
            if (cur_rank < world_size_ - 1 && cur_weight > 0 && cur_weight + w > cur_target) {
                ++cur_rank;
                cur_weight = 0;
                cur_target = target_for(cur_rank);
            }
            assigned_flat[static_cast<size_t>(recv_displs[it.src_rank] + it.src_pos)] = static_cast<int>(cur_rank);
            cur_weight += w;
        }

    }

    std::vector<int> owned_assigned(static_cast<size_t>(local_owned), static_cast<int>(my_rank_));
    MPI_Scatterv(my_rank_ == 0 ? assigned_flat.data() : nullptr,
                 my_rank_ == 0 ? recv_counts.data() : nullptr,
                 my_rank_ == 0 ? recv_displs.data() : nullptr,
                 MPI_INT,
                 owned_assigned.data(),
                 local_owned,
                 MPI_INT,
                 0,
                 comm_);

    std::vector<rank_t> new_owner_rank_per_cell(local_mesh_->n_cells(), my_rank_);
    for (size_t i = 0; i < owned_cells.size(); ++i) {
        new_owner_rank_per_cell[static_cast<size_t>(owned_cells[i])] =
            static_cast<rank_t>(owned_assigned[i]);
    }

    migrate(new_owner_rank_per_cell);

    if (local_mesh_) {
        local_mesh_->event_bus().notify(MeshEvent::PartitionChanged);
    }
	#endif
	}

// ==========================================
// Partition Quality Metrics
// ==========================================

DistributedMesh::PartitionMetrics DistributedMesh::compute_partition_quality() const {
    PartitionMetrics metrics = {};

#ifdef MESH_HAS_MPI
    // =====================================================
    // Load balance metrics
    // =====================================================
    size_t local_n_cells = local_mesh_->n_cells();
    size_t local_n_owned_cells = 0;

    // Count only owned cells (exclude ghosts)
    for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
        if (is_owned_cell(c)) {
            local_n_owned_cells++;
        }
    }

    // Gather cell counts from all ranks
    std::vector<size_t> cells_per_rank(world_size_);
    MPI_Allgather(&local_n_owned_cells, 1, MPI_UNSIGNED_LONG,
                  cells_per_rank.data(), 1, MPI_UNSIGNED_LONG, comm_);

    // Compute load balance statistics
    size_t total_cells = std::accumulate(cells_per_rank.begin(), cells_per_rank.end(), size_t(0));
    metrics.avg_cells_per_rank = total_cells / world_size_;

    auto [min_it, max_it] = std::minmax_element(cells_per_rank.begin(), cells_per_rank.end());
    metrics.min_cells_per_rank = *min_it;
    metrics.max_cells_per_rank = *max_it;

    // Load imbalance factor: (max_load / avg_load) - 1.0
    // Perfect balance = 0.0, higher values indicate worse imbalance
    if (metrics.avg_cells_per_rank > 0) {
        metrics.load_imbalance_factor =
            static_cast<double>(metrics.max_cells_per_rank) / metrics.avg_cells_per_rank - 1.0;
    }

    // =====================================================
    // Communication metrics - edge cuts and shared faces
    // =====================================================
    size_t local_edge_cuts = 0;
    size_t local_shared_faces = 0;

    // Prefer an exact dual edge-cut count when a ghost layer is present (neighbors visible).
    // Fallbacks:
    // - If no ghost layer but faces exist: shared faces correspond to cut edges.
    // - Otherwise: conservative proxy via "boundary cells" (cells that touch a shared vertex).
    const bool can_use_dual_neighbors = (ghost_levels_ > 0);
    if (can_use_dual_neighbors) {
        const auto& gids = local_mesh_->cell_gids();
        for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
            if (!is_owned_cell(c)) continue;
            const gid_t cg = gids[static_cast<size_t>(c)];
            if (cg == INVALID_GID) continue;

            const rank_t cr = owner_rank_cell(c);
            const auto neigh = local_mesh_->cell_neighbors(c);
            for (const auto n : neigh) {
                if (n < 0 || static_cast<size_t>(n) >= local_n_cells) continue;
                if (owner_rank_cell(n) == cr) continue;
                const gid_t ng = gids[static_cast<size_t>(n)];
                if (ng == INVALID_GID) continue;
                // Unique count: enforce a strict ordering by global cell GID.
                if (cg < ng) {
                    ++local_edge_cuts;
                }
            }
        }
    } else if (local_mesh_->n_faces() == 0) {
        for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
            if (!is_owned_cell(c)) continue;
            auto [vertices_ptr, n_verts] = local_mesh_->cell_vertices_span(c);
            bool touches_shared = false;
            for (size_t v = 0; v < n_verts; ++v) {
                const index_t vertex_id = vertices_ptr[v];
                if (is_shared_vertex(vertex_id)) {
                    touches_shared = true;
                    break;
                }
            }
            if (touches_shared) {
                ++local_edge_cuts;
            }
        }
    }

    // Count shared faces separately if faces are available
    if (local_mesh_->n_faces() > 0) {
        for (index_t f = 0; f < static_cast<index_t>(local_mesh_->n_faces()); ++f) {
            if (is_shared_face(f)) {
                local_shared_faces++;
            }
        }
    }

    // Global reduction for communication metrics
    MPI_Allreduce(&local_edge_cuts, &metrics.total_edge_cuts, 1,
                  MPI_UNSIGNED_LONG, MPI_SUM, comm_);
    MPI_Allreduce(&local_shared_faces, &metrics.total_shared_faces, 1,
                  MPI_UNSIGNED_LONG, MPI_SUM, comm_);

    // Each shared face is counted twice (once per rank), so divide by 2.
    metrics.total_shared_faces /= 2;
    // If we could not compute exact dual edge cuts, use shared faces as the best proxy.
    if (!can_use_dual_neighbors && local_mesh_->n_faces() > 0) {
        metrics.total_edge_cuts = metrics.total_shared_faces;
    }

    // =====================================================
    // Ghost cell metrics
    // =====================================================
    size_t local_ghost_cells = ghost_cells_.size();
    MPI_Allreduce(&local_ghost_cells, &metrics.total_ghost_cells, 1,
                  MPI_UNSIGNED_LONG, MPI_SUM, comm_);

    // =====================================================
    // Neighbor connectivity metrics
    // =====================================================
    size_t local_num_neighbors = neighbor_ranks_.size();
    size_t total_neighbors = 0;
    MPI_Allreduce(&local_num_neighbors, &total_neighbors, 1,
                  MPI_UNSIGNED_LONG, MPI_SUM, comm_);
    metrics.avg_neighbors_per_rank = static_cast<double>(total_neighbors) / world_size_;

    // =====================================================
    // Memory footprint metrics
    // =====================================================
    size_t local_memory = 0;

    // Cell storage
    local_memory += local_n_cells * sizeof(CellShape);
    local_memory += local_n_cells * sizeof(gid_t);  // GIDs
    local_memory += local_n_cells * sizeof(label_t);  // Labels

    // Connectivity storage (CSR format)
    local_memory += local_mesh_->cell2vertex_offsets().size() * sizeof(index_t);
    local_memory += local_mesh_->cell2vertex().size() * sizeof(index_t);

    // Vertex storage
    size_t local_n_vertices = local_mesh_->n_vertices();
    local_memory += local_n_vertices * local_mesh_->dim() * sizeof(real_t);  // Coordinates
    local_memory += local_n_vertices * sizeof(gid_t);  // GIDs

    // Face storage (if present)
    if (local_mesh_->n_faces() > 0) {
        size_t local_n_faces = local_mesh_->n_faces();
        local_memory += local_n_faces * sizeof(gid_t);
        // Face connectivity storage estimate (since face2vertex_offsets_ is private)
        // Assuming average 4 vertices per face for 3D meshes
        local_memory += local_n_faces * sizeof(index_t);  // offsets
        local_memory += local_n_faces * 4 * sizeof(index_t);  // indices (estimate)
    }

    // Field data on cells
    auto cell_field_names = local_mesh_->field_names(EntityKind::Volume);
    for (const auto& field_name : cell_field_names) {
        size_t components = local_mesh_->field_components_by_name(EntityKind::Volume, field_name);
        size_t bytes_per_comp = local_mesh_->field_bytes_per_component_by_name(EntityKind::Volume, field_name);
        local_memory += local_n_cells * components * bytes_per_comp;
    }

    // Field data on vertices
    auto vertex_field_names = local_mesh_->field_names(EntityKind::Vertex);
    for (const auto& field_name : vertex_field_names) {
        size_t components = local_mesh_->field_components_by_name(EntityKind::Vertex, field_name);
        size_t bytes_per_comp = local_mesh_->field_bytes_per_component_by_name(EntityKind::Vertex, field_name);
        local_memory += local_n_vertices * components * bytes_per_comp;
    }

    // Distributed mesh overhead
    local_memory += vertex_owner_.size() * sizeof(Ownership);
    local_memory += cell_owner_.size() * sizeof(Ownership);
    local_memory += vertex_owner_rank_.size() * sizeof(rank_t);
    local_memory += cell_owner_rank_.size() * sizeof(rank_t);

    // Gather memory usage from all ranks
    std::vector<size_t> memory_per_rank(world_size_);
    MPI_Allgather(&local_memory, 1, MPI_UNSIGNED_LONG,
                  memory_per_rank.data(), 1, MPI_UNSIGNED_LONG, comm_);

    auto [min_mem_it, max_mem_it] = std::minmax_element(memory_per_rank.begin(), memory_per_rank.end());
    metrics.min_memory_per_rank = *min_mem_it;
    metrics.max_memory_per_rank = *max_mem_it;

    // Memory imbalance factor
    size_t total_memory = std::accumulate(memory_per_rank.begin(), memory_per_rank.end(), size_t(0));
    size_t avg_memory = total_memory / world_size_;
    if (avg_memory > 0) {
        metrics.memory_imbalance_factor =
            static_cast<double>(metrics.max_memory_per_rank) / avg_memory - 1.0;
    }

    // =====================================================
    // Migration metrics (hypothetical - what if we rebalanced?)
    // =====================================================
    // This estimates how many cells would need to migrate for perfect balance
    if (metrics.avg_cells_per_rank > 0) {
        metrics.cells_to_migrate = 0;
        for (size_t rank_cells : cells_per_rank) {
            if (rank_cells > metrics.avg_cells_per_rank) {
                metrics.cells_to_migrate += rank_cells - metrics.avg_cells_per_rank;
            }
        }
    }

    // Estimate migration volume (bytes to transfer)
    if (metrics.cells_to_migrate > 0 && total_cells > 0) {
        // Average bytes per cell
        size_t avg_bytes_per_cell = total_memory / total_cells;
        metrics.migration_volume = metrics.cells_to_migrate * avg_bytes_per_cell;
    }

    // =====================================================
    // Report metrics (optional verbose output)
    // =====================================================
    if (my_rank_ == 0) {
        // This can be enabled with a verbose flag if needed
        #ifdef VERBOSE_PARTITION_METRICS
        std::cout << "\n=== Partition Quality Metrics ===\n";
        std::cout << "Load Balance:\n";
        std::cout << "  Cells per rank: [" << metrics.min_cells_per_rank
                  << ", " << metrics.max_cells_per_rank << "] (avg: "
                  << metrics.avg_cells_per_rank << ")\n";
        std::cout << "  Load imbalance: " << std::fixed << std::setprecision(1)
                  << metrics.load_imbalance_factor * 100 << "%\n";

        std::cout << "Communication:\n";
        std::cout << "  Total edge cuts: " << metrics.total_edge_cuts << "\n";
        std::cout << "  Shared faces: " << metrics.total_shared_faces << "\n";
        std::cout << "  Ghost cells: " << metrics.total_ghost_cells << "\n";
        std::cout << "  Avg neighbors/rank: " << std::fixed << std::setprecision(1)
                  << metrics.avg_neighbors_per_rank << "\n";

        std::cout << "Memory:\n";
        std::cout << "  Memory per rank: [" << metrics.min_memory_per_rank / (1024*1024)
                  << " MB, " << metrics.max_memory_per_rank / (1024*1024) << " MB]\n";
        std::cout << "  Memory imbalance: " << std::fixed << std::setprecision(1)
                  << metrics.memory_imbalance_factor * 100 << "%\n";

        if (metrics.cells_to_migrate > 0) {
            std::cout << "Potential Migration:\n";
            std::cout << "  Cells to migrate: " << metrics.cells_to_migrate << "\n";
            std::cout << "  Migration volume: " << metrics.migration_volume / (1024*1024) << " MB\n";
        }
        std::cout << "================================\n\n";
        #endif
    }

#else
    // Serial case - trivial metrics
    metrics.min_cells_per_rank = metrics.max_cells_per_rank = metrics.avg_cells_per_rank = local_mesh_->n_cells();
    metrics.load_imbalance_factor = 0.0;
    metrics.total_edge_cuts = 0;
    metrics.total_shared_faces = 0;
    metrics.total_ghost_cells = 0;
    metrics.avg_neighbors_per_rank = 0.0;

    // Compute memory for serial case
    size_t total_memory = 0;
    total_memory += local_mesh_->n_cells() * (sizeof(CellShape) + sizeof(gid_t) + sizeof(label_t));
    total_memory += local_mesh_->n_vertices() * local_mesh_->dim() * sizeof(real_t);
    total_memory += local_mesh_->cell2vertex().size() * sizeof(index_t);

    metrics.min_memory_per_rank = metrics.max_memory_per_rank = total_memory;
    metrics.memory_imbalance_factor = 0.0;
    metrics.cells_to_migrate = 0;
    metrics.migration_volume = 0;
#endif

    return metrics;
}

// ==========================================
// Parallel I/O
// ==========================================

DistributedMesh DistributedMesh::load_parallel(const MeshIOOptions& opts, MPI_Comm comm) {
    // =====================================================
    // Parallel Mesh Loading Strategy
    // =====================================================
    // This function implements multiple I/O strategies based on file format:
    // 1. PVTU format: Each rank loads its own piece independently (best scaling)
    // 2. HDF5 format: Collective parallel I/O (requires HDF5 with MPI support)
    // 3. Serial formats: Root loads and distributes (bottleneck for large meshes)
    //
    // Performance characteristics:
    // - PVTU: O(n/p) time, O(n/p) memory per rank, best for large-scale runs
    // - HDF5: O(n/p) time with collective I/O, good for moderate scales
    // - Serial: O(n) time on root, O(n) memory spike on root, limited scalability

#ifdef MESH_HAS_MPI
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Parse file extension and base path
    std::string extension;
    std::string base_path = opts.path;
    size_t dot_pos = opts.path.rfind('.');
    if (dot_pos != std::string::npos) {
        extension = opts.path.substr(dot_pos);
        base_path = opts.path.substr(0, dot_pos);
    }

    // If we need to fall back from parallel meta-formats to a single-file load,
    // use this options object (mutated below when needed).
    MeshIOOptions serial_fallback_opts = opts;

    std::shared_ptr<MeshBase> local_mesh;

    // Start timing for performance monitoring
    double io_start_time = MPI_Wtime();

    // =====================================================
    // Strategy 1: PVTU/PVTK Format (Parallel VTK)
    // =====================================================
    // Expected performance: O(n/p) time, O(n/p) memory
    // Scaling: Near-perfect weak scaling to 10K+ cores
    // Use case: Production runs, large meshes (>10M cells)

    if (extension == ".pvtu" || extension == ".pvtk") {
        // Each rank independently loads its piece (no inter-process comm needed).
        // Prefer parsing the .pvtu to find piece filenames; fall back to naming conventions.
        MeshIOOptions local_opts = opts;
        local_opts.format = "vtu";

        std::string piece_path;
        {
            std::ifstream pvtu_file(opts.path);
            if (pvtu_file.good()) {
                std::string content((std::istreambuf_iterator<char>(pvtu_file)), std::istreambuf_iterator<char>());
                std::vector<std::string> pieces;
                size_t pos = 0;
                while (true) {
                    const size_t ppos = content.find("<Piece", pos);
                    if (ppos == std::string::npos) break;
                    const size_t spos = content.find("Source=\"", ppos);
                    if (spos == std::string::npos) {
                        pos = ppos + 6;
                        continue;
                    }
                    const size_t begin = spos + 8;
                    const size_t end = content.find('"', begin);
                    if (end == std::string::npos) break;
                    pieces.push_back(content.substr(begin, end - begin));
                    pos = end + 1;
                }

                if (static_cast<size_t>(rank) < pieces.size()) {
                    piece_path = pieces[static_cast<size_t>(rank)];
                    // Resolve relative paths against the pvtu directory.
                    if (!piece_path.empty() && piece_path.front() != '/' && opts.path.find('/') != std::string::npos) {
                        const auto slash = opts.path.find_last_of('/');
                        const std::string dir = (slash == std::string::npos) ? std::string() : opts.path.substr(0, slash + 1);
                        piece_path = dir + piece_path;
                    }
                }
            }
        }

        if (piece_path.empty()) {
            // Fallback naming conventions.
            piece_path = base_path + "_p" + std::to_string(rank) + ".vtu";
            std::ifstream test_file(piece_path);
            if (!test_file.good()) {
                piece_path = base_path + "_" + std::to_string(rank) + ".vtu";
            }
        }

        local_opts.path = piece_path;

        std::ifstream test_file(local_opts.path);
        if (!test_file.good()) {
            if (rank == 0) {
                std::cerr << "Warning: PVTU piece not found for rank " << rank
                          << " (expected: " << local_opts.path << "). Falling back to serial load.\n";
            }
            // Force serial fallback (rank 0 loads a single-file VTU and distributes).
            extension = ".vtu";
            serial_fallback_opts = opts;
            serial_fallback_opts.format = "vtu";
            serial_fallback_opts.path = base_path + ".vtu";
        } else {
            test_file.close();

            try {
                local_mesh = std::make_shared<MeshBase>(MeshBase::load(local_opts));

                double io_time = MPI_Wtime() - io_start_time;
                if (rank == 0 && opts.kv.find("verbose") != opts.kv.end()) {
                    std::cout << "PVTU loading completed in " << io_time << " seconds\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "Rank " << rank << " failed to load " << local_opts.path
                          << ": " << e.what() << std::endl;
                MPI_Abort(comm, 1);
            }
        }
    }

    // =====================================================
    // Strategy 2: HDF5 Format (Collective Parallel I/O)
    // =====================================================
    // Expected performance: O(n/p) with collective buffering
    // Scaling: Good to 1K cores with parallel file system
    // Use case: Medium to large meshes on parallel filesystems

    else if (extension == ".h5" || extension == ".hdf5") {
#ifdef MESH_HAS_HDF5_PARALLEL
        // Use HDF5 parallel I/O for efficient collective reading
        // Each rank reads its portion of the mesh simultaneously

        // Set up parallel HDF5 access
        MPI_Info info = MPI_INFO_NULL;
        MPI_Info_create(&info);
        MPI_Info_set(info, "collective_buffering", "true");
        MPI_Info_set(info, "cb_buffer_size", "16777216");  // 16MB buffers

        // Open file collectively
        hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, comm, info);

        hid_t file_id = H5Fopen(opts.path.c_str(), H5F_ACC_RDONLY, plist_id);
        if (file_id < 0) {
            if (rank == 0) {
                std::cerr << "Failed to open HDF5 file: " << opts.path << std::endl;
            }
            MPI_Abort(comm, 1);
        }

        // Read mesh metadata collectively
        hid_t dataset_id = H5Dopen2(file_id, "/mesh/metadata", H5P_DEFAULT);

        // Each rank calculates its portion based on global dimensions
        // ... HDF5 hyperslab selection code ...

        // Collective read with optimized chunking
        hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);

        // Read vertex coordinates, connectivity, fields collectively
        // ... actual HDF5 read operations ...

        H5Pclose(xfer_plist);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        H5Pclose(plist_id);
        MPI_Info_free(&info);

        double io_time = MPI_Wtime() - io_start_time;
        if (rank == 0 && opts.kv.find("verbose") != opts.kv.end()) {
            std::cout << "HDF5 parallel loading completed in " << io_time << " seconds\n";
        }
#else
        if (rank == 0) {
            std::cerr << "HDF5 parallel support not available. Falling back to serial load.\n";
        }
        extension = ".vtu";  // Force serial fallback
#endif
    }

    // =====================================================
    // Strategy 3: Serial Formats (Root loads and distributes)
    // =====================================================
    // Expected performance: O(n) time on root, O(n*log(p)) for distribution
    // Memory: O(n) spike on root, O(n/p) on other ranks after distribution
    // Scaling: Limited to ~100 ranks due to root bottleneck
    // Use case: Small meshes (<1M cells), debugging, legacy formats

    bool used_serial_distribution = false;

    if (!local_mesh) {  // If not already loaded by PVTU or HDF5
        used_serial_distribution = true;
        if (rank == 0) {
            // =====================================================
            // Step 1: Root loads entire mesh (memory bottleneck)
            // =====================================================
            double load_start = MPI_Wtime();
            // Note: opts.path may be a .pvtu/.pvtk; in that case we fall back to a
            // single-file VTU at <base_path>.vtu (see above).
            MeshBase global_mesh = MeshBase::load(serial_fallback_opts);

            if (opts.kv.find("verbose") != opts.kv.end()) {
                std::cout << "Root loaded " << global_mesh.n_cells() << " cells in "
                          << MPI_Wtime() - load_start << " seconds\n";
            }

            // =====================================================
            // Step 2: Partition the mesh for distribution
            // =====================================================
            // Partitioning options:
            // - metis   : Rank-0 METIS graph partitioning (high quality, root bottleneck)
            // - parmetis: Parallel ParMETIS graph partitioning via block pre-distribution + rebalance()
            // - block   : Deterministic contiguous blocks

            size_t n_cells = global_mesh.n_cells();
            std::vector<rank_t> cell_partition(n_cells);

            std::string partition_method = "metis";  // default (falls back to deterministic block when graph is degenerate)
            if (opts.kv.find("partition_method") != opts.kv.end()) {
                partition_method = opts.kv.at("partition_method");
            }

            std::transform(partition_method.begin(), partition_method.end(),
                           partition_method.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

            if (partition_method == "parmetis") {
                // ParMETIS startup partitioning is performed after an initial block distribution.
                // The initial distribution only needs to provide a valid mesh per rank; it does not
                // need to be a good partition.
                cell_partition = partition_cells_block(n_cells, size);
            } else if (partition_method == "block") {
                cell_partition = partition_cells_block(n_cells, size);
            } else {
#if defined(SVMP_HAS_METIS)
                if (partition_method == "metis" || partition_method == "graph") {
                    cell_partition = partition_cells_metis(global_mesh, size, PartitionHint::Cells);
                } else {
                    if (opts.kv.find("verbose") != opts.kv.end()) {
                        std::cerr << "Warning: partition_method='" << partition_method
                                  << "' is not recognized; falling back to METIS partition.\n";
                    }
                    cell_partition = partition_cells_metis(global_mesh, size, PartitionHint::Cells);
                }
#else
                if (opts.kv.find("verbose") != opts.kv.end()) {
                    std::cerr << "Warning: partition_method='" << partition_method
                              << "' requested but METIS support is not enabled; falling back to block partition.\n";
                }
                cell_partition = partition_cells_block(n_cells, size);
#endif
            }

            // =====================================================
            // Step 3: Extract and distribute submeshes
            // =====================================================
            // Current policy: rank 0 extracts each rank's submesh and sends it directly.
            // This is deterministic and correctness-focused; scaling optimizations can be
            // layered in later.

            // Precompute per-rank cell lists so we don't rescan the global mesh for each rank.
            std::vector<size_t> cell_counts(static_cast<size_t>(size), 0);
            for (size_t c = 0; c < n_cells; ++c) {
                const auto r = cell_partition[c];
                if (r < 0 || r >= size) continue;
                cell_counts[static_cast<size_t>(r)]++;
            }

            std::vector<std::vector<index_t>> cells_by_rank(static_cast<size_t>(size));
            for (int r = 0; r < size; ++r) {
                cells_by_rank[static_cast<size_t>(r)].reserve(cell_counts[static_cast<size_t>(r)]);
            }
            for (index_t c = 0; c < static_cast<index_t>(n_cells); ++c) {
                const auto r = cell_partition[static_cast<size_t>(c)];
                if (r < 0 || r >= size) continue;
                cells_by_rank[static_cast<size_t>(r)].push_back(c);
            }

            for (int r = 0; r < size; ++r) {
                // Extract submesh for rank r
                auto submesh = extract_submesh(global_mesh,
                                               cell_partition,
                                               static_cast<rank_t>(r),
                                               &cells_by_rank[static_cast<size_t>(r)]);

                if (r == 0) {
                    local_mesh = std::make_shared<MeshBase>(std::move(submesh));
                } else {
                    // Serialize and send
                    std::vector<char> buffer;
                    serialize_mesh(submesh, buffer);

                    int buffer_size = static_cast<int>(buffer.size());
                    MPI_Send(&buffer_size, 1, MPI_INT, r, 0, comm);
                    if (buffer_size > 0) {
                        MPI_Send(buffer.data(), buffer_size, MPI_CHAR, r, 1, comm);
                    }
                }
            }

        } else {
            // =====================================================
            // Non-root ranks: Receive their submesh
            // =====================================================

            // Direct receive from root
            int buffer_size;
            MPI_Recv(&buffer_size, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);

            std::vector<char> buffer(static_cast<size_t>(buffer_size));
            if (buffer_size > 0) {
                MPI_Recv(buffer.data(), buffer_size, MPI_CHAR, 0, 1, comm, MPI_STATUS_IGNORE);
            }

            local_mesh = std::make_shared<MeshBase>();
            deserialize_mesh(buffer, *local_mesh);
        }

        double dist_time = MPI_Wtime() - io_start_time;
        if (rank == 0 && opts.kv.find("verbose") != opts.kv.end()) {
            std::cout << "Serial load + distribution completed in " << dist_time << " seconds\n";
        }
    }

    // =====================================================
    // Step 4: Set up distributed mesh structure
    // =====================================================
    // Create the distributed mesh wrapper
    DistributedMesh dmesh(local_mesh, comm);

    // =====================================================
    // Optional: Parallel ParMETIS repartitioning after initial distribution
    // =====================================================
    // For serial input formats, we can obtain a scalable initial partition by:
    //   1) Root loads + block-distributes a valid initial mesh to all ranks.
    //   2) All ranks call ParMETIS on the distributed cell dual graph and migrate cells.
    //
    // This avoids rank-0 partitioning work and provides a startup path comparable to the legacy solver.
    std::string requested_partition_method = "metis";
    if (opts.kv.find("partition_method") != opts.kv.end()) {
        requested_partition_method = opts.kv.at("partition_method");
        std::transform(requested_partition_method.begin(), requested_partition_method.end(),
                       requested_partition_method.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    }

#if defined(SVMP_HAS_PARMETIS)
    if (used_serial_distribution && requested_partition_method == "parmetis") {
        dmesh.rebalance(PartitionHint::ParMetis, opts.kv);
    }
#else
    if (used_serial_distribution && requested_partition_method == "parmetis") {
        if (rank == 0 && opts.kv.find("verbose") != opts.kv.end()) {
            std::cerr << "Warning: partition_method='parmetis' requested but ParMETIS is not enabled; "
                         "falling back to rank-0 METIS/block partition.\n";
        }
    }
#endif

    // =====================================================
    // Step 5: Interpret VTK ghost metadata (if present)
    // =====================================================
    // When loading from pre-partitioned VTU pieces, the files may already
    // include ghost/duplicate entities. Preserve that information by mapping
    // the conventional VTK array "vtkGhostType" to ownership flags.
    //
    // This is independent of VTK being enabled at compile time: the array is
    // read as a MeshBase field and is interpreted here.
    if (dmesh.local_mesh_) {
        constexpr const char* kGhostName = "vtkGhostType";

        auto apply_ghost_field = [&](EntityKind kind,
                                     std::vector<Ownership>& owners,
                                     std::vector<rank_t>& owner_ranks) {
            if (!dmesh.local_mesh_->has_field(kind, kGhostName)) {
                return;
            }

            const auto h = dmesh.local_mesh_->field_handle(kind, kGhostName);
            if (h.id == 0) {
                return;
            }
            if (dmesh.local_mesh_->field_components(h) != 1) {
                return;
            }

            const size_t n = dmesh.local_mesh_->field_entity_count(h);
            if (owners.size() != n) {
                owners.assign(n, Ownership::Owned);
                owner_ranks.assign(n, dmesh.my_rank_);
            }

            const auto t = dmesh.local_mesh_->field_type(h);
            if (t == FieldScalarType::UInt8) {
                const auto* ghost = dmesh.local_mesh_->field_data_as<std::uint8_t>(h);
                if (!ghost) {
                    return;
                }
                for (index_t i = 0; i < static_cast<index_t>(n); ++i) {
                    const bool is_ghost = (ghost[static_cast<size_t>(i)] != 0);
                    owners[static_cast<size_t>(i)] = is_ghost ? Ownership::Ghost : Ownership::Owned;
                    if (is_ghost) {
                        owner_ranks[static_cast<size_t>(i)] = -1;
                    }
                }
                return;
            }

            // Best-effort: accept Int32/Int64 ghost arrays by treating any nonzero as ghost.
            if (t == FieldScalarType::Int32) {
                const auto* ghost = dmesh.local_mesh_->field_data_as<std::int32_t>(h);
                if (!ghost) {
                    return;
                }
                for (index_t i = 0; i < static_cast<index_t>(n); ++i) {
                    const bool is_ghost = (ghost[static_cast<size_t>(i)] != 0);
                    owners[static_cast<size_t>(i)] = is_ghost ? Ownership::Ghost : Ownership::Owned;
                    if (is_ghost) {
                        owner_ranks[static_cast<size_t>(i)] = -1;
                    }
                }
                return;
            }

            if (t == FieldScalarType::Int64) {
                const auto* ghost = dmesh.local_mesh_->field_data_as<std::int64_t>(h);
                if (!ghost) {
                    return;
                }
                for (index_t i = 0; i < static_cast<index_t>(n); ++i) {
                    const bool is_ghost = (ghost[static_cast<size_t>(i)] != 0);
                    owners[static_cast<size_t>(i)] = is_ghost ? Ownership::Ghost : Ownership::Owned;
                    if (is_ghost) {
                        owner_ranks[static_cast<size_t>(i)] = -1;
                    }
                }
                return;
            }
        };

        apply_ghost_field(EntityKind::Vertex, dmesh.vertex_owner_, dmesh.vertex_owner_rank_);
        apply_ghost_field(EntityKind::Volume, dmesh.cell_owner_, dmesh.cell_owner_rank_);
    }

    // =====================================================
    // Step 6: Build ghost layers for communication
    // =====================================================
    // Expected time: O(ghost_layers * neighbors * boundary_size)
    // Memory: O(ghost_cells) additional storage

    int ghost_layers = 0;  // Default (explicit ghosting)
    if (opts.kv.find("ghost_layers") != opts.kv.end()) {
        ghost_layers = std::stoi(opts.kv.at("ghost_layers"));
    }
    if (ghost_layers > 0) {
        dmesh.build_ghost_layer(ghost_layers);
    } else {
        // Establish owned/shared ownership and exchange patterns on the base partition.
        dmesh.build_exchange_patterns();
    }

    // =====================================================
    // Performance Summary
    // =====================================================
    double total_time = MPI_Wtime() - io_start_time;
    double max_time;
    MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0 && opts.kv.find("verbose") != opts.kv.end()) {
        std::cout << "\n=== Parallel Load Performance ===\n";
        std::cout << "Total time: " << max_time << " seconds\n";
        std::cout << "Mesh size: " << dmesh.global_n_cells() << " cells, "
                  << dmesh.global_n_vertices() << " vertices\n";
        std::cout << "Ranks: " << size << "\n";
        std::cout << "Cells/rank: " << dmesh.global_n_cells() / size << " average\n";

        // Compute and report load balance
        auto metrics = dmesh.compute_partition_quality();
        std::cout << "Load imbalance: " << std::fixed << std::setprecision(1)
                  << metrics.load_imbalance_factor * 100 << "%\n";
        std::cout << "Ghost cells: " << metrics.total_ghost_cells << " total\n";
        std::cout << "==================================\n\n";
    }

    return dmesh;
#else
    // Serial fallback - no MPI available
    auto mesh = std::make_shared<MeshBase>(MeshBase::load(opts));
    return DistributedMesh(mesh, comm);
#endif
}

void DistributedMesh::save_parallel(const MeshIOOptions& opts) const {
#ifdef MESH_HAS_MPI
    if (world_size_ == 1) {
        // Serial save
        local_mesh_->save(opts);
        return;
    }

    // Check if the path ends with .pvtu - if so, use PVTU format
    bool use_pvtu = false;
    std::string base_path = opts.path;
    std::string extension;

    size_t dot_pos = opts.path.rfind('.');
    if (dot_pos != std::string::npos) {
        extension = opts.path.substr(dot_pos);
        base_path = opts.path.substr(0, dot_pos);

        // Check for .pvtu or .pvtk extensions
        if (extension == ".pvtu" || extension == ".pvtk") {
            use_pvtu = true;
        }
    }

    if (use_pvtu) {
        // PVTU format: Each rank saves a .vtu file, rank 0 writes master .pvtu file

        // Step 1: Each rank writes its piece as a .vtu file
        MeshIOOptions local_opts = opts;
        local_opts.format = "vtu";  // Force VTU format
        local_opts.path = base_path + "_p" + std::to_string(my_rank_) + ".vtu";

        // Attach standard VTK ghost metadata so parallel readers can de-duplicate.
        // We use VTK's conventional array name "vtkGhostType" with UInt8 values:
        //   0 -> normal, 1 -> duplicate (ghost)
        const char* kGhostName = "vtkGhostType";
        bool created_cell_ghost = false;
        bool created_point_ghost = false;
        FieldHandle cell_ghost_h;
        FieldHandle point_ghost_h;

        if (!local_mesh_->has_field(EntityKind::Volume, kGhostName)) {
            cell_ghost_h = local_mesh_->attach_field(EntityKind::Volume, kGhostName, FieldScalarType::UInt8, 1);
            created_cell_ghost = true;
        } else {
            cell_ghost_h = local_mesh_->field_handle(EntityKind::Volume, kGhostName);
        }
        if (!local_mesh_->has_field(EntityKind::Vertex, kGhostName)) {
            point_ghost_h = local_mesh_->attach_field(EntityKind::Vertex, kGhostName, FieldScalarType::UInt8, 1);
            created_point_ghost = true;
        } else {
            point_ghost_h = local_mesh_->field_handle(EntityKind::Vertex, kGhostName);
        }

        if (cell_ghost_h.id != 0) {
            auto* ghost = local_mesh_->field_data_as<std::uint8_t>(cell_ghost_h);
            if (ghost) {
                for (index_t c = 0; c < static_cast<index_t>(local_mesh_->n_cells()); ++c) {
                    ghost[static_cast<size_t>(c)] = is_ghost_cell(c) ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
                }
            }
        }

        if (point_ghost_h.id != 0) {
            auto* ghost = local_mesh_->field_data_as<std::uint8_t>(point_ghost_h);
            if (ghost) {
                for (index_t v = 0; v < static_cast<index_t>(local_mesh_->n_vertices()); ++v) {
                    ghost[static_cast<size_t>(v)] = is_ghost_vertex(v) ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
                }
            }
        }

        // Write this rank's piece (owned + ghosts, with ghost metadata).
        local_mesh_->save(local_opts);

        // Step 2: Rank 0 writes the master .pvtu file
        if (my_rank_ == 0) {
            std::ofstream pvtu(opts.path);
            if (!pvtu.is_open()) {
                throw std::runtime_error("Failed to open PVTU file: " + opts.path);
            }

            pvtu << "<?xml version=\"1.0\"?>\n";
            pvtu << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
            pvtu << "  <PUnstructuredGrid GhostLevel=\"" << ghost_levels_ << "\">\n";

            // Write PPoints and PPointData sections (metadata about what's in the pieces)
            pvtu << "    <PPoints>\n";
            pvtu << "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\"/>\n";
            pvtu << "    </PPoints>\n";

            // -----------------------------------------------------------------
            // Metadata about arrays present in the piece files.
            //
            // NOTE: ParaView/VTK parallel readers rely on <PPointData>/<PCellData>
            // to advertise arrays without opening a piece.
            // -----------------------------------------------------------------

            auto vertex_fields = local_mesh_->field_names(EntityKind::Vertex);
            pvtu << "    <PPointData>\n";
            // Global IDs are always written by VTKWriter::write_global_ids().
            pvtu << "      <PDataArray type=\"Int64\" Name=\"GlobalVertexID\" NumberOfComponents=\"1\" format=\"ascii\"/>\n";
            for (const auto& field_name : vertex_fields) {
                auto type = local_mesh_->field_type_by_name(EntityKind::Vertex, field_name);
                auto components = local_mesh_->field_components_by_name(EntityKind::Vertex, field_name);

                std::string vtk_type = "Float64";  // Default
                if (type == FieldScalarType::Float32) vtk_type = "Float32";
                else if (type == FieldScalarType::Int32) vtk_type = "Int32";
                else if (type == FieldScalarType::Int64) vtk_type = "Int64";
                else if (type == FieldScalarType::UInt8) vtk_type = "UInt8";

                pvtu << "      <PDataArray type=\"" << vtk_type
                     << "\" Name=\"" << field_name
                     << "\" NumberOfComponents=\"" << components
                     << "\" format=\"ascii\"/>\n";
            }
            pvtu << "    </PPointData>\n";

            auto cell_fields = local_mesh_->field_names(EntityKind::Volume);
            pvtu << "    <PCellData>\n";
            // Region labels are always written by VTKWriter::write_region_labels().
            pvtu << "      <PDataArray type=\"Int32\" Name=\"RegionID\" NumberOfComponents=\"1\" format=\"ascii\"/>\n";
            // Global IDs are always written by VTKWriter::write_global_ids().
            pvtu << "      <PDataArray type=\"Int64\" Name=\"GlobalCellID\" NumberOfComponents=\"1\" format=\"ascii\"/>\n";
            for (const auto& field_name : cell_fields) {
                auto type = local_mesh_->field_type_by_name(EntityKind::Volume, field_name);
                auto components = local_mesh_->field_components_by_name(EntityKind::Volume, field_name);

                std::string vtk_type = "Float64";  // Default
                if (type == FieldScalarType::Float32) vtk_type = "Float32";
                else if (type == FieldScalarType::Int32) vtk_type = "Int32";
                else if (type == FieldScalarType::Int64) vtk_type = "Int64";
                else if (type == FieldScalarType::UInt8) vtk_type = "UInt8";

                pvtu << "      <PDataArray type=\"" << vtk_type
                     << "\" Name=\"" << field_name
                     << "\" NumberOfComponents=\"" << components
                     << "\" format=\"ascii\"/>\n";
            }
            pvtu << "    </PCellData>\n";

            // Write piece references
            // Piece paths in a .pvtu should be relative to the .pvtu location.
            // Our piece files are written to `<base_path>_p<r>.vtu`; here we only
            // advertise the filenames so readers (including our own `load_parallel`)
            // can resolve them against the .pvtu directory.
            std::string base_name = base_path;
            {
                const auto slash = base_name.find_last_of("/\\");
                if (slash != std::string::npos) {
                    base_name = base_name.substr(slash + 1);
                }
            }
            for (int r = 0; r < world_size_; ++r) {
                pvtu << "    <Piece Source=\"" << base_name << "_p" << r << ".vtu\"/>\n";
            }

            pvtu << "  </PUnstructuredGrid>\n";
            pvtu << "</VTKFile>\n";
            pvtu.close();
        }

        // Remove temporary arrays to keep the runtime mesh state unchanged.
        // Rank 0 delays removal until after writing the master file so it can
        // advertise the ghost arrays in <PPointData>/<PCellData>.
        if (created_cell_ghost && cell_ghost_h.id != 0) {
            local_mesh_->remove_field(cell_ghost_h);
        }
        if (created_point_ghost && point_ghost_h.id != 0) {
            local_mesh_->remove_field(point_ghost_h);
        }

        // Ensure all ranks finish writing before returning
        MPI_Barrier(comm_);

    } else {
        // Original behavior: Each rank saves its own file with rank suffix
        MeshIOOptions local_opts = opts;
        local_opts.path = opts.path + "_rank" + std::to_string(my_rank_);
        local_mesh_->save(local_opts);
    }
#else
    local_mesh_->save(opts);
#endif
}

// ==========================================
// Global Reductions
// ==========================================

size_t DistributedMesh::global_n_vertices() const {
#ifdef MESH_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized || comm_ == MPI_COMM_NULL || world_size_ <= 1) {
        return local_mesh_->n_vertices();
    }

    // Count only owned vertices
    size_t local_owned = 0;
    for (size_t i = 0; i < vertex_owner_.size(); ++i) {
        if (vertex_owner_[i] == Ownership::Owned) {
            local_owned++;
        }
    }

    size_t global_total = 0;
    MPI_Allreduce(&local_owned, &global_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_);
    return global_total;
#else
    return local_mesh_->n_vertices();
#endif
}

std::pair<real_t,real_t> DistributedMesh::global_quality_range(const std::string& metric) const {
#ifdef MESH_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized || comm_ == MPI_COMM_NULL || world_size_ <= 1) {
        // Serial / single-rank path: "global" means over the local mesh.
        return local_mesh_->global_quality_range(metric);
    }

    // Compute range over owned cells only to avoid double-counting ghosts.
    real_t local_min = 1e300;
    real_t local_max = -1e300;

    const index_t n_cells = static_cast<index_t>(local_mesh_->n_cells());
    for (index_t c = 0; c < n_cells; ++c) {
        if (!is_owned_cell(c)) {
            continue;
        }
        const real_t q = local_mesh_->compute_quality(c, metric);
        local_min = std::min(local_min, q);
        local_max = std::max(local_max, q);
    }

    real_t global_min = 0.0;
    real_t global_max = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm_);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm_);

    return {global_min, global_max};
#else
    return local_mesh_->global_quality_range(metric);
#endif
}

size_t DistributedMesh::global_n_cells() const {
#ifdef MESH_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized || comm_ == MPI_COMM_NULL || world_size_ <= 1) {
        return local_mesh_->n_cells();
    }

    // Count only owned cells
    size_t local_owned = 0;
    for (size_t i = 0; i < cell_owner_.size(); ++i) {
        if (cell_owner_[i] == Ownership::Owned) {
            local_owned++;
        }
    }

    size_t global_total = 0;
    MPI_Allreduce(&local_owned, &global_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_);
    return global_total;
#else
    return local_mesh_->n_cells();
#endif
}

size_t DistributedMesh::global_n_faces() const {
#ifdef MESH_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized || comm_ == MPI_COMM_NULL || world_size_ <= 1) {
        return local_mesh_->n_faces();
    }

    // Count only owned faces
    size_t local_owned = 0;
    for (size_t i = 0; i < face_owner_.size(); ++i) {
        if (face_owner_[i] == Ownership::Owned) {
            local_owned++;
        }
    }

    size_t global_total = 0;
    MPI_Allreduce(&local_owned, &global_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_);
    return global_total;
#else
    return local_mesh_->n_faces();
#endif
}

BoundingBox DistributedMesh::global_bounding_box() const {
#ifdef MESH_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized || comm_ == MPI_COMM_NULL || world_size_ <= 1) {
        return local_mesh_->bounding_box();
    }

    auto local_bbox = local_mesh_->bounding_box();

    BoundingBox global_bbox;
    MPI_Allreduce(local_bbox.min.data(), global_bbox.min.data(), 3,
                 MPI_DOUBLE, MPI_MIN, comm_);
    MPI_Allreduce(local_bbox.max.data(), global_bbox.max.data(), 3,
                 MPI_DOUBLE, MPI_MAX, comm_);

    return global_bbox;
#else
    return local_mesh_->bounding_box();
#endif
}

// ==========================================
// Distributed Search
// ==========================================

PointLocateResult DistributedMesh::locate_point_global(
    const std::array<real_t,3>& x, Configuration cfg) const {

#ifdef MESH_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized || comm_ == MPI_COMM_NULL) {
        return local_mesh_->locate_point(x, cfg);
    }
    int comm_rank = 0;
    int comm_size = 1;
    MPI_Comm_rank(comm_, &comm_rank);
    MPI_Comm_size(comm_, &comm_size);
    if (comm_size <= 1) {
        return local_mesh_->locate_point(x, cfg);
    }

    // All ranks must participate in the same collective sequence.
    // Each rank attempts a local point location, then we select a canonical "winner"
    // (prefer owned > shared > ghost, then lowest rank) and report:
    // - On the winning rank: the local locate_point result (cell_id is a local index)
    // - On other ranks: found==true, cell_id==INVALID_INDEX (point exists globally but not locally)
    const auto local_result = local_mesh_->locate_point(x, cfg);

    int local_priority = 3; // not found
    if (local_result.found) {
        local_priority = 0; // default to owned if ownership is unavailable
        if (local_result.cell_id >= 0 &&
            static_cast<size_t>(local_result.cell_id) < cell_owner_.size()) {
            switch (cell_owner_[static_cast<size_t>(local_result.cell_id)]) {
                case Ownership::Owned:   local_priority = 0; break;
                case Ownership::Shared:  local_priority = 1; break;
                case Ownership::Ghost:   local_priority = 2; break;
            }
        }
    }

    // Encode selection key: smaller is better; UINT64_MAX means "not found".
    // [ priority (upper 32 bits) | rank (lower 32 bits) ]
    constexpr uint64_t kNotFoundKey = std::numeric_limits<uint64_t>::max();
    const uint64_t local_key =
        local_result.found
            ? ((static_cast<uint64_t>(static_cast<uint32_t>(local_priority)) << 32) |
               static_cast<uint32_t>(comm_rank))
            : kNotFoundKey;

    uint64_t global_key = 0;
    MPI_Datatype u64_type =
#ifdef MPI_UINT64_T
        MPI_UINT64_T;
#else
        MPI_UNSIGNED_LONG_LONG;
#endif
    MPI_Allreduce(&local_key, &global_key, 1, u64_type, MPI_MIN, comm_);

    if (global_key == kNotFoundKey) {
        return PointLocateResult{};
    }

    const rank_t winner_rank = static_cast<rank_t>(static_cast<uint32_t>(global_key & 0xffffffffULL));
    if (winner_rank == static_cast<rank_t>(comm_rank)) {
        return local_result;
    }

    PointLocateResult result;
    result.found = true;
    result.cell_id = INVALID_INDEX;
    return result;
#else
    return local_mesh_->locate_point(x, cfg);
#endif
}

std::vector<PointLocateResult> DistributedMesh::locate_points_global(
    const std::vector<std::array<real_t,3>>& X, Configuration cfg) const {

#ifdef MESH_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized || comm_ == MPI_COMM_NULL) {
        return local_mesh_->locate_points(X, cfg);
    }
    int comm_rank = 0;
    int comm_size = 1;
    MPI_Comm_rank(comm_, &comm_rank);
    MPI_Comm_size(comm_, &comm_size);
    if (comm_size <= 1) {
        return local_mesh_->locate_points(X, cfg);
    }

    const auto local_results = local_mesh_->locate_points(X, cfg);

    constexpr uint64_t kNotFoundKey = std::numeric_limits<uint64_t>::max();
    std::vector<uint64_t> local_keys(X.size(), kNotFoundKey);

    for (size_t i = 0; i < local_results.size(); ++i) {
        const auto& lr = local_results[i];
        if (!lr.found) {
            continue;
        }

        int local_priority = 0; // default to owned if ownership is unavailable
        if (lr.cell_id >= 0 && static_cast<size_t>(lr.cell_id) < cell_owner_.size()) {
            switch (cell_owner_[static_cast<size_t>(lr.cell_id)]) {
                case Ownership::Owned:   local_priority = 0; break;
                case Ownership::Shared:  local_priority = 1; break;
                case Ownership::Ghost:   local_priority = 2; break;
            }
        }

        local_keys[i] =
            (static_cast<uint64_t>(static_cast<uint32_t>(local_priority)) << 32) |
            static_cast<uint32_t>(comm_rank);
    }

    std::vector<uint64_t> global_keys(X.size(), kNotFoundKey);
    MPI_Datatype u64_type =
#ifdef MPI_UINT64_T
        MPI_UINT64_T;
#else
        MPI_UNSIGNED_LONG_LONG;
#endif
    if (!global_keys.empty()) {
        MPI_Allreduce(local_keys.data(), global_keys.data(), static_cast<int>(global_keys.size()),
                      u64_type, MPI_MIN, comm_);
    }

    std::vector<PointLocateResult> out;
    out.reserve(X.size());
    for (size_t i = 0; i < global_keys.size(); ++i) {
        const uint64_t gk = global_keys[i];
        if (gk == kNotFoundKey) {
            out.push_back(PointLocateResult{});
            continue;
        }

        const rank_t winner_rank = static_cast<rank_t>(static_cast<uint32_t>(gk & 0xffffffffULL));
        if (winner_rank == static_cast<rank_t>(comm_rank)) {
            out.push_back(local_results[i]);
        } else {
            PointLocateResult r;
            r.found = true;
            r.cell_id = INVALID_INDEX;
            out.push_back(r);
        }
    }

    return out;
#else
    return local_mesh_->locate_points(X, cfg);
#endif
}

// ==========================================
// Communication Patterns
// ==========================================

void DistributedMesh::build_exchange_patterns() {
#ifdef MESH_HAS_MPI
    if (world_size_ == 1) {
        return;
    }

    // Ensure derived entity GIDs are canonical before using them for shared/ghost detection or exchange patterns.
    if (local_mesh_) {
        ensure_canonical_face_gids(*local_mesh_);
        ensure_canonical_edge_gids(*local_mesh_);
    }

    // Guardrail: if every non-empty rank is still using the default local-iota GIDs
    // (0..n-1), shared detection will be incorrect because global IDs will collide
    // across ranks. This often happens when users call build_from_arrays(...) on each
    // rank but forget to assign consistent global IDs.
    if (local_mesh_) {
        const int local_has_vertices = local_mesh_->n_vertices() > 0 ? 1 : 0;
        const int local_has_cells = local_mesh_->n_cells() > 0 ? 1 : 0;

        const int local_vertices_iota = gids_are_local_iota(local_mesh_->vertex_gids()) ? 1 : 0;
        const int local_cells_iota = gids_are_local_iota(local_mesh_->cell_gids()) ? 1 : 0;

        int ranks_with_vertices = 0;
        int ranks_with_cells = 0;
        MPI_Allreduce(&local_has_vertices, &ranks_with_vertices, 1, MPI_INT, MPI_SUM, comm_);
        MPI_Allreduce(&local_has_cells, &ranks_with_cells, 1, MPI_INT, MPI_SUM, comm_);

        const int local_vertices_iota_for_reduce = local_has_vertices ? local_vertices_iota : 1;
        const int local_cells_iota_for_reduce = local_has_cells ? local_cells_iota : 1;

        int all_vertices_iota = 1;
        int all_cells_iota = 1;
        MPI_Allreduce(&local_vertices_iota_for_reduce, &all_vertices_iota, 1, MPI_INT, MPI_MIN, comm_);
        MPI_Allreduce(&local_cells_iota_for_reduce, &all_cells_iota, 1, MPI_INT, MPI_MIN, comm_);

        if ((ranks_with_vertices > 1 && all_vertices_iota == 1) ||
            (ranks_with_cells > 1 && all_cells_iota == 1)) {
            throw std::runtime_error(
                "DistributedMesh::build_exchange_patterns: detected default local GIDs on all ranks. "
                "Assign globally consistent IDs with MeshBase::set_vertex_gids/set_cell_gids, "
                "or build via load_mesh(..., MeshComm::world()) / build_from_arrays_global_and_partition().");
        }
    }

    // If no ghost layer exists, detect shared entities first so ownership arrays
    // reflect the base partition before building exchange patterns.
    if (ghost_levels_ == 0 && ghost_vertices_.empty() && ghost_cells_.empty() && ghost_faces_.empty() && ghost_edges_.empty()) {
        // Preserve any explicitly-marked ghosts (e.g., in tests) while still
        // allowing gather_shared_entities() to recompute canonical owner ranks.
        std::vector<index_t> explicit_ghost_vertices;
        std::vector<index_t> explicit_ghost_cells;
        std::vector<index_t> explicit_ghost_faces;
        std::vector<index_t> explicit_ghost_edges;

        explicit_ghost_vertices.reserve(vertex_owner_.size());
        for (index_t v = 0; v < static_cast<index_t>(vertex_owner_.size()); ++v) {
            if (vertex_owner_[static_cast<size_t>(v)] == Ownership::Ghost) {
                explicit_ghost_vertices.push_back(v);
            }
        }

        explicit_ghost_cells.reserve(cell_owner_.size());
        for (index_t c = 0; c < static_cast<index_t>(cell_owner_.size()); ++c) {
            if (cell_owner_[static_cast<size_t>(c)] == Ownership::Ghost) {
                explicit_ghost_cells.push_back(c);
            }
        }

        explicit_ghost_faces.reserve(face_owner_.size());
        for (index_t f = 0; f < static_cast<index_t>(face_owner_.size()); ++f) {
            if (face_owner_[static_cast<size_t>(f)] == Ownership::Ghost) {
                explicit_ghost_faces.push_back(f);
            }
        }

        explicit_ghost_edges.reserve(edge_owner_.size());
        for (index_t e = 0; e < static_cast<index_t>(edge_owner_.size()); ++e) {
            if (edge_owner_[static_cast<size_t>(e)] == Ownership::Ghost) {
                explicit_ghost_edges.push_back(e);
            }
        }

        gather_shared_entities();

        for (index_t v : explicit_ghost_vertices) {
            vertex_owner_[static_cast<size_t>(v)] = Ownership::Ghost;
        }
        for (index_t c : explicit_ghost_cells) {
            cell_owner_[static_cast<size_t>(c)] = Ownership::Ghost;
        }
        for (index_t f : explicit_ghost_faces) {
            face_owner_[static_cast<size_t>(f)] = Ownership::Ghost;
        }
        for (index_t e : explicit_ghost_edges) {
            edge_owner_[static_cast<size_t>(e)] = Ownership::Ghost;
        }
    }

    // Clear existing patterns
    vertex_exchange_ = ExchangePattern{};
    cell_exchange_ = ExchangePattern{};
    face_exchange_ = ExchangePattern{};
    edge_exchange_ = ExchangePattern{};
    neighbor_ranks_.clear();

    auto build_request_response_pattern =
        [&](const std::vector<Ownership>& ownership,
            const std::vector<rank_t>& owner_rank,
            const std::vector<gid_t>& gids,
            const auto& global_to_local) -> ExchangePattern {
            ExchangePattern pattern;

            // Build "request" lists: for each owner rank, send the GIDs we need values for.
            std::vector<std::vector<gid_t>> request_gids(static_cast<size_t>(world_size_));
            std::vector<std::vector<index_t>> recv_lists(static_cast<size_t>(world_size_));

            for (index_t i = 0; i < static_cast<index_t>(ownership.size()); ++i) {
                if (ownership[static_cast<size_t>(i)] == Ownership::Owned) {
                    continue;
                }
                const rank_t owner = owner_rank[static_cast<size_t>(i)];
                if (owner < 0 || owner == my_rank_ || owner >= world_size_) {
                    continue;
                }
                request_gids[static_cast<size_t>(owner)].push_back(gids[static_cast<size_t>(i)]);
                recv_lists[static_cast<size_t>(owner)].push_back(i);
            }

            // Exchange request GIDs using MPI_Alltoallv (bytes).
            std::vector<int> send_counts(world_size_, 0);
            std::vector<int> send_displs(world_size_ + 1, 0);
            for (int r = 0; r < world_size_; ++r) {
                send_counts[r] = static_cast<int>(request_gids[static_cast<size_t>(r)].size() * sizeof(gid_t));
                send_displs[r + 1] = send_displs[r] + send_counts[r];
            }

            std::vector<char> send_buffer(static_cast<size_t>(send_displs[world_size_]));
            for (int r = 0; r < world_size_; ++r) {
                const auto& vec = request_gids[static_cast<size_t>(r)];
                if (!vec.empty()) {
                    std::memcpy(send_buffer.data() + send_displs[r],
                                vec.data(),
                                vec.size() * sizeof(gid_t));
                }
            }

            std::vector<int> recv_counts(world_size_, 0);
            MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                         recv_counts.data(), 1, MPI_INT, comm_);

            std::vector<int> recv_displs(world_size_ + 1, 0);
            for (int r = 0; r < world_size_; ++r) {
                recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
            }

            std::vector<char> recv_buffer(static_cast<size_t>(recv_displs[world_size_]));

            MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_BYTE,
                          recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_BYTE,
                          comm_);

            // Receive side: map the entities we requested from each owner rank.
            for (int r = 0; r < world_size_; ++r) {
                if (!recv_lists[static_cast<size_t>(r)].empty()) {
                    pattern.recv_ranks.push_back(r);
                    pattern.recv_lists.push_back(std::move(recv_lists[static_cast<size_t>(r)]));
                }
            }

            // Send side: for each requesting rank, build the local index list (same order as requests).
            for (int r = 0; r < world_size_; ++r) {
                const int n_bytes = recv_counts[r];
                if (n_bytes <= 0) {
                    continue;
                }

                const int n_gids = n_bytes / static_cast<int>(sizeof(gid_t));
                const gid_t* requested =
                    reinterpret_cast<const gid_t*>(recv_buffer.data() + recv_displs[r]);

                std::vector<index_t> send_list;
                send_list.reserve(static_cast<size_t>(n_gids));
                for (int i = 0; i < n_gids; ++i) {
                    send_list.push_back(global_to_local(requested[i]));
                }

                pattern.send_ranks.push_back(r);
                pattern.send_lists.push_back(std::move(send_list));
            }

            return pattern;
        };

    vertex_exchange_ =
        build_request_response_pattern(
            vertex_owner_,
            vertex_owner_rank_,
            local_mesh_->vertex_gids(),
            [&](gid_t gid) { return local_mesh_->global_to_local_vertex(gid); });

    cell_exchange_ =
        build_request_response_pattern(
            cell_owner_,
            cell_owner_rank_,
            local_mesh_->cell_gids(),
            [&](gid_t gid) { return local_mesh_->global_to_local_cell(gid); });

    face_exchange_ =
        build_request_response_pattern(
            face_owner_,
            face_owner_rank_,
            local_mesh_->face_gids(),
            [&](gid_t gid) { return local_mesh_->global_to_local_face(gid); });

    edge_exchange_ =
        build_request_response_pattern(
            edge_owner_,
            edge_owner_rank_,
            local_mesh_->edge_gids(),
            [&](gid_t gid) { return local_mesh_->global_to_local_edge(gid); });

    // Update neighbor ranks based on the final patterns.
    auto add_neighbors = [&](const ExchangePattern& p) {
        for (rank_t r : p.send_ranks) {
            if (r >= 0 && r < world_size_ && r != my_rank_) neighbor_ranks_.insert(r);
        }
        for (rank_t r : p.recv_ranks) {
            if (r >= 0 && r < world_size_ && r != my_rank_) neighbor_ranks_.insert(r);
        }
    };
    add_neighbors(vertex_exchange_);
    add_neighbors(cell_exchange_);
    add_neighbors(face_exchange_);
    add_neighbors(edge_exchange_);

    if (local_mesh_) {
        local_mesh_->event_bus().notify(MeshEvent::PartitionChanged);
    }
#endif
}

// ==========================================
// Helper Methods
// ==========================================

void DistributedMesh::exchange_entity_data(EntityKind kind, const void* send_data,
                                          void* recv_data, size_t bytes_per_entity,
                                          const ExchangePattern& pattern) {
#ifdef MESH_HAS_MPI
    if (world_size_ == 1 || (pattern.send_ranks.empty() && pattern.recv_ranks.empty())) {
        return;
    }

    const uint8_t* send_bytes = static_cast<const uint8_t*>(send_data);
    uint8_t* recv_bytes = static_cast<uint8_t*>(recv_data);

    size_t n_entities = 0;
    switch (kind) {
        case EntityKind::Vertex: n_entities = local_mesh_->n_vertices(); break;
        case EntityKind::Volume: n_entities = local_mesh_->n_cells(); break;
        case EntityKind::Face:   n_entities = local_mesh_->n_faces(); break;
        case EntityKind::Edge:   n_entities = local_mesh_->n_edges(); break;
    }

    // Allocate buffers
    std::vector<std::vector<uint8_t>> send_buffers(pattern.send_ranks.size());
    std::vector<std::vector<uint8_t>> recv_buffers(pattern.recv_ranks.size());

    // Pack send buffers
    for (size_t i = 0; i < pattern.send_ranks.size(); ++i) {
        const auto& send_list = pattern.send_lists[i];
        send_buffers[i].resize(send_list.size() * bytes_per_entity);

        for (size_t j = 0; j < send_list.size(); ++j) {
            index_t entity = send_list[j];
            uint8_t* out = &send_buffers[i][j * bytes_per_entity];
            if (entity < 0 || static_cast<size_t>(entity) >= n_entities) {
                std::memset(out, 0, bytes_per_entity);
            } else {
                std::memcpy(out,
                            &send_bytes[static_cast<size_t>(entity) * bytes_per_entity],
                            bytes_per_entity);
            }
        }
    }

    // Size recv buffers
    for (size_t i = 0; i < pattern.recv_ranks.size(); ++i) {
        const auto& recv_list = pattern.recv_lists[i];
        recv_buffers[i].resize(recv_list.size() * bytes_per_entity);
    }

    // Non-blocking sends
    std::vector<MPI_Request> requests;
    requests.reserve(pattern.send_ranks.size() + pattern.recv_ranks.size());

    for (size_t i = 0; i < pattern.send_ranks.size(); ++i) {
        MPI_Request req;
        MPI_Isend(send_buffers[i].data(),
                 static_cast<int>(send_buffers[i].size()),
                 MPI_BYTE, pattern.send_ranks[i], 0, comm_, &req);
        requests.push_back(req);
    }

    // Non-blocking receives
    for (size_t i = 0; i < pattern.recv_ranks.size(); ++i) {
        MPI_Request req;
        MPI_Irecv(recv_buffers[i].data(),
                 static_cast<int>(recv_buffers[i].size()),
                 MPI_BYTE, pattern.recv_ranks[i], 0, comm_, &req);
        requests.push_back(req);
    }

    // Wait for all communication
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    // Unpack recv buffers
    for (size_t i = 0; i < pattern.recv_ranks.size(); ++i) {
        const auto& recv_list = pattern.recv_lists[i];

        for (size_t j = 0; j < recv_list.size(); ++j) {
            index_t entity = recv_list[j];
            if (entity < 0 || static_cast<size_t>(entity) >= n_entities) {
                continue;
            }
            std::memcpy(&recv_bytes[static_cast<size_t>(entity) * bytes_per_entity],
                        &recv_buffers[i][j * bytes_per_entity],
                        bytes_per_entity);
        }
    }
#endif
}

void DistributedMesh::gather_shared_entities() {
    // =====================================================
    // Scalable Shared Entity Detection Algorithm
    // =====================================================
    // Purpose: Identify entities that exist on multiple ranks
    // Algorithm: Hash-based distribution with point-to-point communication
    //
    // Performance characteristics:
    // - Time: O(n_entities/p + n_shared * log(p))
    // - Memory: O(n_entities/p) distributed hash table
    // - Communication: O(p) messages total, O(n_shared) data volume
    // - Scaling: Near-linear to 10K+ cores
    //
    // Algorithm overview:
    // 1. Each GID is assigned to a "home" rank via hash function
    // 2. Entities send their GIDs to home ranks
    // 3. Home ranks identify duplicates (shared entities)
    // 4. Home ranks notify owners about shared status
    // 5. Apply deterministic ownership rule (lowest rank wins)

#ifdef MESH_HAS_MPI
    if (world_size_ == 1) {
        return;  // No shared entities in serial
    }

    double gather_start = MPI_Wtime();

    // Get local entity GIDs
    const auto& vertex_gids = local_mesh_->vertex_gids();
    const auto& cell_gids = local_mesh_->cell_gids();
    ensure_canonical_face_gids(*local_mesh_);
    const auto& face_gids = local_mesh_->face_gids();
    ensure_canonical_edge_gids(*local_mesh_);
    const auto& edge_gids = local_mesh_->edge_gids();

    // =====================================================
    // Phase 1: Hash-based GID distribution
    // =====================================================
    // Each GID is assigned to a home rank: hash(gid) % world_size
    // This distributes the ownership detection workload evenly

    // Hash function for GID -> home rank assignment
    auto gid_home_rank = [this](gid_t gid, EntityKind kind) -> rank_t {
        // Use a good hash to ensure even distribution
        // MurmurHash-inspired mixing
        uint64_t h = static_cast<uint64_t>(gid);
        h ^= static_cast<uint64_t>(static_cast<int>(kind)) * 0x9e3779b97f4a7c15ULL;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        return static_cast<rank_t>(h % world_size_);
    };

    // =====================================================
    // Step 1: Sort local GIDs by home rank
    // =====================================================
    // Time: O(n_entities * log(p)) for sorting by destination
    // This enables efficient packing for communication

    struct GIDInfo {
        gid_t gid;
        EntityKind kind;
        index_t local_id;
        rank_t source_rank;
        int8_t is_ghost;  // 1 if this rank's copy is a ghost
    };

    // Collect all local GIDs with their info
    std::vector<std::vector<GIDInfo>> gids_by_dest(world_size_);

    auto is_local_ghost = [](const std::vector<Ownership>& own, index_t id) -> int8_t {
        if (id < 0) return 0;
        if (static_cast<size_t>(id) >= own.size()) return 0;
        return (own[static_cast<size_t>(id)] == Ownership::Ghost) ? 1 : 0;
    };

    // Edge ghostness: prefer identifying edges that exist only on ghost cells.
    std::vector<int8_t> edge_is_ghost(edge_gids.size(), 0);
    if (!edge_gids.empty() && cell_owner_.size() == local_mesh_->n_cells()) {
        bool any_ghost_cell = false;
        for (const auto& o : cell_owner_) {
            if (o == Ownership::Ghost) {
                any_ghost_cell = true;
                break;
            }
        }

        if (any_ghost_cell && local_mesh_->n_edges() > 0) {
            std::vector<offset_t> edge2cell_offsets;
            std::vector<index_t> edge2cell;
            MeshTopology::build_edge2cell(*local_mesh_, local_mesh_->edge2vertex(), edge2cell_offsets, edge2cell);

            for (index_t e = 0; e < static_cast<index_t>(edge_is_ghost.size()); ++e) {
                const offset_t start = edge2cell_offsets[static_cast<size_t>(e)];
                const offset_t end = edge2cell_offsets[static_cast<size_t>(e + 1)];
                if (start >= end) {
                    continue;
                }

                bool has_non_ghost = false;
                for (offset_t off = start; off < end; ++off) {
                    const index_t c = edge2cell[static_cast<size_t>(off)];
                    if (c < 0 || static_cast<size_t>(c) >= cell_owner_.size()) continue;
                    if (cell_owner_[static_cast<size_t>(c)] != Ownership::Ghost) {
                        has_non_ghost = true;
                        break;
                    }
                }
                if (!has_non_ghost) {
                    edge_is_ghost[static_cast<size_t>(e)] = 1;
                }
            }
        }
    }

    // Respect explicitly marked edge ghosts even if the mesh has no ghost cells (tests/advanced use).
    if (edge_owner_.size() == edge_is_ghost.size()) {
        for (size_t e = 0; e < edge_is_ghost.size(); ++e) {
            if (edge_owner_[e] == Ownership::Ghost) edge_is_ghost[e] = 1;
        }
    }

    // Add vertices
    for (index_t v = 0; v < static_cast<index_t>(vertex_gids.size()); ++v) {
        gid_t gid = vertex_gids[v];
        if (gid >= 0) {  // Valid GID
            rank_t home = gid_home_rank(gid, EntityKind::Vertex);
            gids_by_dest[home].push_back({gid, EntityKind::Vertex, v, my_rank_, is_local_ghost(vertex_owner_, v)});
        }
    }

    // Add cells
    for (index_t c = 0; c < static_cast<index_t>(cell_gids.size()); ++c) {
        gid_t gid = cell_gids[c];
        if (gid >= 0) {  // Valid GID
            rank_t home = gid_home_rank(gid, EntityKind::Volume);
            gids_by_dest[home].push_back({gid, EntityKind::Volume, c, my_rank_, is_local_ghost(cell_owner_, c)});
        }
    }

    // Add faces (if available)
    for (index_t f = 0; f < static_cast<index_t>(face_gids.size()); ++f) {
        gid_t gid = face_gids[f];
        if (gid >= 0) {  // Valid GID
            rank_t home = gid_home_rank(gid, EntityKind::Face);
            gids_by_dest[home].push_back({gid, EntityKind::Face, f, my_rank_, is_local_ghost(face_owner_, f)});
        }
    }

    // Add edges (if available)
    for (index_t e = 0; e < static_cast<index_t>(edge_gids.size()); ++e) {
        gid_t gid = edge_gids[e];
        if (gid >= 0) {  // Valid GID
            rank_t home = gid_home_rank(gid, EntityKind::Edge);
            gids_by_dest[home].push_back({gid, EntityKind::Edge, e, my_rank_, edge_is_ghost[static_cast<size_t>(e)]});
        }
    }

    // =====================================================
    // Step 2: Exchange GID info with home ranks
    // =====================================================
    // Time: O(n_entities/p) average per rank
    // Communication: All-to-all personalized pattern

    // Prepare send counts and displacements
    std::vector<int> send_counts(world_size_);
    std::vector<int> send_displs(world_size_ + 1, 0);

    for (int r = 0; r < world_size_; ++r) {
        send_counts[r] = static_cast<int>(gids_by_dest[r].size() * sizeof(GIDInfo));
        send_displs[r + 1] = send_displs[r] + send_counts[r];
    }

    // Flatten send buffer
    std::vector<char> send_buffer(send_displs[world_size_]);
    for (int r = 0; r < world_size_; ++r) {
        if (!gids_by_dest[r].empty()) {
            std::memcpy(send_buffer.data() + send_displs[r],
                       gids_by_dest[r].data(),
                       gids_by_dest[r].size() * sizeof(GIDInfo));
        }
    }

    // Exchange counts using MPI_Alltoall
    std::vector<int> recv_counts(world_size_);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                recv_counts.data(), 1, MPI_INT, comm_);

    // Prepare receive buffer
    std::vector<int> recv_displs(world_size_ + 1, 0);
    for (int r = 0; r < world_size_; ++r) {
        recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
    }
    std::vector<char> recv_buffer(recv_displs[world_size_]);

    // Exchange GID information
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_BYTE,
                  recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_BYTE,
                  comm_);

    // =====================================================
    // Step 3: Identify shared entities at home ranks
    // =====================================================
    // Time: O(n_received) with hash table operations
    // Each home rank identifies GIDs that appear multiple times

    struct KindGID {
        gid_t gid;
        EntityKind kind;
    };

    struct KindGIDLess {
        bool operator()(const KindGID& a, const KindGID& b) const noexcept {
            const int ak = static_cast<int>(a.kind);
            const int bk = static_cast<int>(b.kind);
            if (ak != bk) return ak < bk;
            return a.gid < b.gid;
        }
    };

    struct RankOcc {
        rank_t rank;
        int8_t is_ghost;
    };

    // Track ranks per (kind,gid) key to avoid collisions between entity types, and carry
    // whether each contributing copy is a ghost (ghost copies must not "win" ownership).
    std::map<KindGID, std::vector<RankOcc>, KindGIDLess> gid_owners;

    // Process received GIDs
    for (int r = 0; r < world_size_; ++r) {
        int n_infos = recv_counts[r] / sizeof(GIDInfo);
        GIDInfo* infos = reinterpret_cast<GIDInfo*>(recv_buffer.data() + recv_displs[r]);

        for (int i = 0; i < n_infos; ++i) {
            gid_owners[{infos[i].gid, infos[i].kind}].push_back({infos[i].source_rank, infos[i].is_ghost});
        }
    }

    // Identify shared GIDs and determine owners
    struct SharedInfo {
        gid_t gid;
        EntityKind kind;
        rank_t owner;  // Lowest rank owning this GID
        bool is_shared;
    };

    std::map<rank_t, std::vector<SharedInfo>> shared_info_by_rank;

    for (const auto& [key, owners] : gid_owners) {
        // De-duplicate ranks (a rank can contribute the same key multiple times if its GIDs are not unique).
        auto uniq = owners;
        std::sort(uniq.begin(), uniq.end(),
                  [](const RankOcc& a, const RankOcc& b) { return a.rank < b.rank; });
        uniq.erase(std::unique(uniq.begin(), uniq.end(),
                               [](const RankOcc& a, const RankOcc& b) { return a.rank == b.rank; }),
                   uniq.end());

        if (uniq.empty()) continue;

        // Ownership rule:
        // - If any non-ghost copy exists, the lowest-rank *non-ghost* copy owns the entity.
        // - Otherwise (all copies are ghost), fall back to lowest rank.
        rank_t min_any = uniq[0].rank;
        rank_t min_non_ghost = world_size_;
        bool have_non_ghost = false;
        for (const auto& occ : uniq) {
            min_any = std::min(min_any, occ.rank);
            if (!occ.is_ghost) {
                have_non_ghost = true;
                min_non_ghost = std::min(min_non_ghost, occ.rank);
            }
        }
        const rank_t owner_rank = have_non_ghost ? min_non_ghost : min_any;
        const bool is_shared = (uniq.size() > 1);

        for (const auto& occ : uniq) {
            shared_info_by_rank[occ.rank].push_back({key.gid, key.kind, owner_rank, is_shared});
        }
    }

    // =====================================================
    // Step 4: Send shared status back to source ranks
    // =====================================================
    // Time: O(n_shared) for packing/unpacking
    // Communication: Reverse all-to-all pattern

    // Prepare reply data
    send_buffer.clear();
    send_counts.assign(world_size_, 0);

    for (int r = 0; r < world_size_; ++r) {
        if (shared_info_by_rank.count(r) > 0) {
            send_counts[r] = static_cast<int>(shared_info_by_rank[r].size() * sizeof(SharedInfo));
        }
    }

    // Compute displacements
    send_displs[0] = 0;
    for (int r = 0; r < world_size_; ++r) {
        send_displs[r + 1] = send_displs[r] + send_counts[r];
    }

    // Pack send buffer
    send_buffer.resize(send_displs[world_size_]);
    for (int r = 0; r < world_size_; ++r) {
        if (shared_info_by_rank.count(r) > 0) {
            std::memcpy(send_buffer.data() + send_displs[r],
                       shared_info_by_rank[r].data(),
                       shared_info_by_rank[r].size() * sizeof(SharedInfo));
        }
    }

    // Exchange counts
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                recv_counts.data(), 1, MPI_INT, comm_);

    // Prepare receive buffer
    recv_displs[0] = 0;
    for (int r = 0; r < world_size_; ++r) {
        recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
    }
    recv_buffer.resize(recv_displs[world_size_]);

    // Exchange shared information
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_BYTE,
                  recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_BYTE,
                  comm_);

    // =====================================================
    // Step 5: Process received shared status
    // =====================================================
    // Time: O(n_entities) to update ownership arrays
    // Apply ownership rules based on shared status

    // Build GID to local index maps for fast lookup
    std::unordered_map<gid_t, index_t> vertex_gid_to_local;
    std::unordered_map<gid_t, index_t> cell_gid_to_local;
    std::unordered_map<gid_t, index_t> face_gid_to_local;
    std::unordered_map<gid_t, index_t> edge_gid_to_local;

    for (index_t v = 0; v < static_cast<index_t>(vertex_gids.size()); ++v) {
        vertex_gid_to_local[vertex_gids[v]] = v;
    }
    for (index_t c = 0; c < static_cast<index_t>(cell_gids.size()); ++c) {
        cell_gid_to_local[cell_gids[c]] = c;
    }
    for (index_t f = 0; f < static_cast<index_t>(face_gids.size()); ++f) {
        face_gid_to_local[face_gids[f]] = f;
    }
    for (index_t e = 0; e < static_cast<index_t>(edge_gids.size()); ++e) {
        edge_gid_to_local[edge_gids[e]] = e;
    }

    // Initialize ownership arrays if not already done
    if (vertex_owner_.size() != vertex_gids.size()) {
        vertex_owner_.resize(vertex_gids.size(), Ownership::Owned);
        vertex_owner_rank_.resize(vertex_gids.size(), my_rank_);
    }
    if (cell_owner_.size() != cell_gids.size()) {
        cell_owner_.resize(cell_gids.size(), Ownership::Owned);
        cell_owner_rank_.resize(cell_gids.size(), my_rank_);
    }
    if (face_owner_.size() != face_gids.size()) {
        face_owner_.resize(face_gids.size(), Ownership::Owned);
        face_owner_rank_.resize(face_gids.size(), my_rank_);
    }
    if (edge_owner_.size() != edge_gids.size()) {
        edge_owner_.resize(edge_gids.size(), Ownership::Owned);
        edge_owner_rank_.resize(edge_gids.size(), my_rank_);
    }

    // Process shared information
    for (int r = 0; r < world_size_; ++r) {
        int n_infos = recv_counts[r] / sizeof(SharedInfo);
        SharedInfo* infos = reinterpret_cast<SharedInfo*>(recv_buffer.data() + recv_displs[r]);

        for (int i = 0; i < n_infos; ++i) {
            const SharedInfo& info = infos[i];

            switch (info.kind) {
                case EntityKind::Vertex: {
                    auto it = vertex_gid_to_local.find(info.gid);
                    if (it != vertex_gid_to_local.end()) {
                        index_t local_id = it->second;
                        vertex_owner_rank_[local_id] = info.owner;

                        if (info.is_shared) {
                            if (info.owner == my_rank_) {
                                vertex_owner_[local_id] = Ownership::Owned;
                            } else {
                                vertex_owner_[local_id] = Ownership::Shared;
                            }
                        } else {
                            vertex_owner_[local_id] = Ownership::Owned;
                        }
                    }
                    break;
                }
                case EntityKind::Volume: {
                    auto it = cell_gid_to_local.find(info.gid);
                    if (it != cell_gid_to_local.end()) {
                        index_t local_id = it->second;
                        cell_owner_rank_[local_id] = info.owner;

                        if (info.is_shared) {
                            if (info.owner == my_rank_) {
                                cell_owner_[local_id] = Ownership::Owned;
                            } else {
                                cell_owner_[local_id] = Ownership::Shared;
                            }
                        } else {
                            cell_owner_[local_id] = Ownership::Owned;
                        }
                    }
                    break;
                }
                case EntityKind::Face: {
                    auto it = face_gid_to_local.find(info.gid);
                    if (it != face_gid_to_local.end()) {
                        index_t local_id = it->second;
                        face_owner_rank_[local_id] = info.owner;

                        if (info.is_shared) {
                            if (info.owner == my_rank_) {
                                face_owner_[local_id] = Ownership::Owned;
                            } else {
                                face_owner_[local_id] = Ownership::Shared;
                            }
                        } else {
                            face_owner_[local_id] = Ownership::Owned;
                        }
                    }
                    break;
                }
                case EntityKind::Edge: {
                    auto it = edge_gid_to_local.find(info.gid);
                    if (it != edge_gid_to_local.end()) {
                        index_t local_id = it->second;
                        edge_owner_rank_[local_id] = info.owner;

                        if (info.is_shared) {
                            if (info.owner == my_rank_) {
                                edge_owner_[local_id] = Ownership::Owned;
                            } else {
                                edge_owner_[local_id] = Ownership::Shared;
                            }
                        } else {
                            edge_owner_[local_id] = Ownership::Owned;
                        }
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }

    // =====================================================
    // Step 6: Build neighbor ranks set
    // =====================================================
    // Collect unique neighbor ranks for future communication
    neighbor_ranks_.clear();

    for (const auto& owner : vertex_owner_rank_) {
        if (owner != my_rank_) {
            neighbor_ranks_.insert(owner);
        }
    }
    for (const auto& owner : cell_owner_rank_) {
        if (owner != my_rank_) {
            neighbor_ranks_.insert(owner);
        }
    }

    // =====================================================
    // Performance reporting (optional)
    // =====================================================
    double gather_time = MPI_Wtime() - gather_start;
    double max_gather_time;
    MPI_Reduce(&gather_time, &max_gather_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);

    const bool verbose = (getenv("MESH_VERBOSE") != nullptr);
    if (verbose) {
        // Count total shared entities (collective on all ranks).
        int local_shared_verts = 0;
        int local_shared_cells = 0;
        for (const auto& ownership : vertex_owner_) {
            if (ownership == Ownership::Shared) local_shared_verts++;
        }
        for (const auto& ownership : cell_owner_) {
            if (ownership == Ownership::Shared) local_shared_cells++;
        }

        int total_shared_verts = 0;
        int total_shared_cells = 0;
        MPI_Reduce(&local_shared_verts, &total_shared_verts, 1, MPI_INT, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_shared_cells, &total_shared_cells, 1, MPI_INT, MPI_SUM, 0, comm_);

        if (my_rank_ == 0) {
            std::cout << "\n=== Shared Entity Detection ===\n";
            std::cout << "Time: " << max_gather_time << " seconds\n";
            std::cout << "Shared vertices: " << total_shared_verts << "\n";
            std::cout << "Shared cells: " << total_shared_cells << "\n";
            std::cout << "Avg neighbors/rank: " << neighbor_ranks_.size() << "\n";
            std::cout << "===============================\n\n";
        }
    }
#endif
}

void DistributedMesh::sync_ghost_metadata() {
    // =====================================================
    // Ghost Metadata Synchronization Algorithm
    // =====================================================
    // Purpose: Ensure consistent ownership and field data across ranks
    // Algorithm: Two-phase neighbor communication with consensus protocol
    //
    // Performance characteristics:
    // - Time: O(n_ghosts + n_neighbors * message_size)
    // - Memory: O(n_ghosts * metadata_size)
    // - Communication: 2 * n_neighbors messages (non-blocking)
    // - Scaling: O(log P) neighbors typical for good partitions
    //
    // Phase 1: Exchange ownership metadata (current implementation)
    // Phase 2: Synchronize field data for shared/ghost entities

#ifdef MESH_HAS_MPI
    if (world_size_ == 1 || (ghost_vertices_.empty() && ghost_cells_.empty())) {
        return;  // Nothing to synchronize
    }

    double sync_start = MPI_Wtime();

    // =====================================================
    // Phase 1: Exchange ownership information
    // =====================================================
    // This ensures all ranks agree on who owns what entity
    // Uses a deterministic consensus rule: lowest rank wins

    // Step 1: Identify neighbor ranks
    // Expected: O(n_entities) time, O(n_neighbors) space
    // Typical n_neighbors = O(log P) for well-partitioned meshes
    std::set<rank_t> potential_neighbors;

    for (index_t n = 0; n < static_cast<index_t>(vertex_owner_.size()); ++n) {
        if (vertex_owner_[n] == Ownership::Ghost || vertex_owner_[n] == Ownership::Shared) {
            rank_t owner = vertex_owner_rank_[n];
            if (owner >= 0 && owner != my_rank_) {
                potential_neighbors.insert(owner);
            }
        }
    }

    for (index_t c = 0; c < static_cast<index_t>(cell_owner_.size()); ++c) {
        if (cell_owner_[c] == Ownership::Ghost || cell_owner_[c] == Ownership::Shared) {
            rank_t owner = cell_owner_rank_[c];
            if (owner >= 0 && owner != my_rank_) {
                potential_neighbors.insert(owner);
            }
        }
    }

    std::vector<rank_t> neighbors(potential_neighbors.begin(), potential_neighbors.end());
    int n_neighbors = static_cast<int>(neighbors.size());

    if (n_neighbors == 0) {
        return;  // No neighbors to synchronize with
    }

    // Step 2: Prepare metadata for exchange
    // For each neighbor, pack:
    // - Number of ghost vertices/cells we have from them
    // - Global IDs of those entities
    // - Current ownership status and owner rank

    struct EntityMetadata {
        gid_t global_id;
        int8_t ownership;  // Cast from Ownership enum
        rank_t owner_rank;
    };

    const auto& vertex_gids = local_mesh_->vertex_gids();
    const auto& cell_gids = local_mesh_->cell_gids();

    // Build per-neighbor metadata lists
    std::vector<std::vector<EntityMetadata>> vertex_metadata_to_send(n_neighbors);
    std::vector<std::vector<EntityMetadata>> cell_metadata_to_send(n_neighbors);

    for (int i = 0; i < n_neighbors; ++i) {
        rank_t neighbor = neighbors[i];

        // Collect vertex metadata
        for (index_t n = 0; n < static_cast<index_t>(vertex_owner_.size()); ++n) {
            if ((vertex_owner_[n] == Ownership::Ghost || vertex_owner_[n] == Ownership::Shared)
                && vertex_owner_rank_[n] == neighbor) {
                EntityMetadata meta;
                meta.global_id = vertex_gids[n];
                meta.ownership = static_cast<int8_t>(vertex_owner_[n]);
                meta.owner_rank = vertex_owner_rank_[n];
                vertex_metadata_to_send[i].push_back(meta);
            }
        }

        // Collect cell metadata
        for (index_t c = 0; c < static_cast<index_t>(cell_owner_.size()); ++c) {
            if ((cell_owner_[c] == Ownership::Ghost || cell_owner_[c] == Ownership::Shared)
                && cell_owner_rank_[c] == neighbor) {
                EntityMetadata meta;
                meta.global_id = cell_gids[c];
                meta.ownership = static_cast<int8_t>(cell_owner_[c]);
                meta.owner_rank = cell_owner_rank_[c];
                cell_metadata_to_send[i].push_back(meta);
            }
        }
    }

    // Step 3: Exchange metadata counts with neighbors
    std::vector<int> vertex_send_counts(n_neighbors);
    std::vector<int> cell_send_counts(n_neighbors);
    std::vector<int> vertex_recv_counts(n_neighbors);
    std::vector<int> cell_recv_counts(n_neighbors);

    for (int i = 0; i < n_neighbors; ++i) {
        vertex_send_counts[i] = static_cast<int>(vertex_metadata_to_send[i].size());
        cell_send_counts[i] = static_cast<int>(cell_metadata_to_send[i].size());
    }

    // Exchange counts using point-to-point communication
    std::vector<MPI_Request> count_requests;
    count_requests.reserve(n_neighbors * 4);

    for (int i = 0; i < n_neighbors; ++i) {
        MPI_Request req;

        // Send vertex count
        MPI_Isend(&vertex_send_counts[i], 1, MPI_INT, neighbors[i], 100, comm_, &req);
        count_requests.push_back(req);

        // Receive vertex count
        MPI_Irecv(&vertex_recv_counts[i], 1, MPI_INT, neighbors[i], 100, comm_, &req);
        count_requests.push_back(req);

        // Send cell count
        MPI_Isend(&cell_send_counts[i], 1, MPI_INT, neighbors[i], 101, comm_, &req);
        count_requests.push_back(req);

        // Receive cell count
        MPI_Irecv(&cell_recv_counts[i], 1, MPI_INT, neighbors[i], 101, comm_, &req);
        count_requests.push_back(req);
    }

    MPI_Waitall(static_cast<int>(count_requests.size()), count_requests.data(), MPI_STATUSES_IGNORE);

    // Step 4: Allocate receive buffers
    std::vector<std::vector<EntityMetadata>> vertex_metadata_to_recv(n_neighbors);
    std::vector<std::vector<EntityMetadata>> cell_metadata_to_recv(n_neighbors);

    for (int i = 0; i < n_neighbors; ++i) {
        vertex_metadata_to_recv[i].resize(vertex_recv_counts[i]);
        cell_metadata_to_recv[i].resize(cell_recv_counts[i]);
    }

    // Step 5: Exchange metadata using non-blocking communication
    std::vector<MPI_Request> data_requests;
    data_requests.reserve(n_neighbors * 4);

    for (int i = 0; i < n_neighbors; ++i) {
        MPI_Request req;

        // Send vertex metadata
        if (vertex_send_counts[i] > 0) {
            MPI_Isend(vertex_metadata_to_send[i].data(),
                     vertex_send_counts[i] * sizeof(EntityMetadata),
                     MPI_BYTE, neighbors[i], 200, comm_, &req);
            data_requests.push_back(req);
        }

        // Receive vertex metadata
        if (vertex_recv_counts[i] > 0) {
            MPI_Irecv(vertex_metadata_to_recv[i].data(),
                     vertex_recv_counts[i] * sizeof(EntityMetadata),
                     MPI_BYTE, neighbors[i], 200, comm_, &req);
            data_requests.push_back(req);
        }

        // Send cell metadata
        if (cell_send_counts[i] > 0) {
            MPI_Isend(cell_metadata_to_send[i].data(),
                     cell_send_counts[i] * sizeof(EntityMetadata),
                     MPI_BYTE, neighbors[i], 201, comm_, &req);
            data_requests.push_back(req);
        }

        // Receive cell metadata
        if (cell_recv_counts[i] > 0) {
            MPI_Irecv(cell_metadata_to_recv[i].data(),
                     cell_recv_counts[i] * sizeof(EntityMetadata),
                     MPI_BYTE, neighbors[i], 201, comm_, &req);
            data_requests.push_back(req);
        }
    }

    MPI_Waitall(static_cast<int>(data_requests.size()), data_requests.data(), MPI_STATUSES_IGNORE);

    // Step 6: Establish ownership consensus
    // Rule: For each entity, the rank with the LOWEST rank number is the canonical owner
    // This ensures deterministic, consistent ownership across all ranks

    // Build maps from global ID to local index
    std::map<gid_t, index_t> gid_to_vertex;
    std::map<gid_t, index_t> gid_to_cell;

    for (index_t n = 0; n < static_cast<index_t>(vertex_gids.size()); ++n) {
        gid_to_vertex[vertex_gids[n]] = n;
    }

    for (index_t c = 0; c < static_cast<index_t>(cell_gids.size()); ++c) {
        gid_to_cell[cell_gids[c]] = c;
    }

    // Process received vertex metadata and apply consensus
    for (int i = 0; i < n_neighbors; ++i) {
        for (const auto& meta : vertex_metadata_to_recv[i]) {
            auto it = gid_to_vertex.find(meta.global_id);
            if (it != gid_to_vertex.end()) {
                index_t local_idx = it->second;

                // Apply ownership consensus rule
                // If neighbor claims ownership and has lower rank, accept it
                rank_t neighbor_owner = meta.owner_rank;
                rank_t current_owner = vertex_owner_rank_[local_idx];

                // Lowest rank wins ownership
                if (neighbor_owner < current_owner) {
                    vertex_owner_rank_[local_idx] = neighbor_owner;
                }

                // Update ownership status based on consensus
                if (vertex_owner_rank_[local_idx] == my_rank_) {
                    vertex_owner_[local_idx] = Ownership::Owned;
                } else if (meta.ownership == static_cast<int8_t>(Ownership::Shared) ||
                          vertex_owner_[local_idx] == Ownership::Shared) {
                    vertex_owner_[local_idx] = Ownership::Shared;
                } else {
                    vertex_owner_[local_idx] = Ownership::Ghost;
                }
            }
        }
    }

    // Process received cell metadata and apply consensus
    for (int i = 0; i < n_neighbors; ++i) {
        for (const auto& meta : cell_metadata_to_recv[i]) {
            auto it = gid_to_cell.find(meta.global_id);
            if (it != gid_to_cell.end()) {
                index_t local_idx = it->second;

                // Apply ownership consensus rule
                rank_t neighbor_owner = meta.owner_rank;
                rank_t current_owner = cell_owner_rank_[local_idx];

                // Lowest rank wins ownership
                if (neighbor_owner < current_owner) {
                    cell_owner_rank_[local_idx] = neighbor_owner;
                }

                // Update ownership status based on consensus
                if (cell_owner_rank_[local_idx] == my_rank_) {
                    cell_owner_[local_idx] = Ownership::Owned;
                } else if (meta.ownership == static_cast<int8_t>(Ownership::Shared) ||
                          cell_owner_[local_idx] == Ownership::Shared) {
                    cell_owner_[local_idx] = Ownership::Shared;
                } else {
                    cell_owner_[local_idx] = Ownership::Ghost;
                }
            }
        }
    }

    // Step 7: Update face ownership based on cell ownership
    // Faces inherit ownership from their adjacent cells
    // Rule: Face is owned by the rank that owns its lower-numbered adjacent cell
    // Time: O(n_faces)
    for (index_t f = 0; f < static_cast<index_t>(face_owner_.size()); ++f) {
        // Get adjacent cells for this face
        if (f < static_cast<index_t>(local_mesh_->n_faces())) {
            auto face_cells = local_mesh_->face_cells(f);
            index_t cell0 = face_cells[0];
            index_t cell1 = face_cells[1];

            // Determine face ownership based on adjacent cells
            if (cell0 != INVALID_INDEX && cell0 < static_cast<index_t>(cell_owner_.size())) {
                // Face inherits from first adjacent cell
                if (cell_owner_[cell0] == Ownership::Owned) {
                    face_owner_[f] = Ownership::Owned;
                    face_owner_rank_[f] = my_rank_;
                } else if (cell_owner_[cell0] == Ownership::Shared) {
                    face_owner_[f] = Ownership::Shared;
                    face_owner_rank_[f] = cell_owner_rank_[cell0];
                } else {
                    face_owner_[f] = Ownership::Ghost;
                    face_owner_rank_[f] = cell_owner_rank_[cell0];
                }
            } else if (cell1 != INVALID_INDEX && cell1 < static_cast<index_t>(cell_owner_.size())) {
                // Boundary face - owned by the rank owning the adjacent cell
                face_owner_[f] = cell_owner_[cell1];
                face_owner_rank_[f] = cell_owner_rank_[cell1];
            }
        }
    }

    // =====================================================
    // Phase 2: Synchronize field data for ghost entities
    // =====================================================
    // This ensures ghost copies have up-to-date field values
    // Uses the exchange patterns built earlier for efficiency

    // Synchronize vertex fields
    auto vertex_field_names = local_mesh_->field_names(EntityKind::Vertex);
    for (const auto& field_name : vertex_field_names) {
        synchronize_field_data(EntityKind::Vertex, field_name);
    }

    // Synchronize cell fields
    auto cell_field_names = local_mesh_->field_names(EntityKind::Volume);
    for (const auto& field_name : cell_field_names) {
        synchronize_field_data(EntityKind::Volume, field_name);
    }

    // =====================================================
    // Phase 3: Update neighbor set for future communication
    // =====================================================
    // Store the final set of neighbors for optimized future exchanges
    neighbor_ranks_.clear();
    for (index_t v = 0; v < static_cast<index_t>(vertex_owner_.size()); ++v) {
        if (vertex_owner_[v] == Ownership::Ghost || vertex_owner_[v] == Ownership::Shared) {
            neighbor_ranks_.insert(vertex_owner_rank_[v]);
        }
    }
    for (index_t c = 0; c < static_cast<index_t>(cell_owner_.size()); ++c) {
        if (cell_owner_[c] == Ownership::Ghost || cell_owner_[c] == Ownership::Shared) {
            neighbor_ranks_.insert(cell_owner_rank_[c]);
        }
    }

    // =====================================================
    // Performance reporting (optional)
    // =====================================================
    double sync_time = MPI_Wtime() - sync_start;
    double max_sync_time;
    MPI_Reduce(&sync_time, &max_sync_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);

    if (my_rank_ == 0 && getenv("MESH_VERBOSE")) {
        std::cout << "Ghost metadata sync completed in " << max_sync_time << " seconds\n";
        std::cout << "  Neighbors: " << neighbor_ranks_.size() << " average\n";
        std::cout << "  Ghost vertices: " << ghost_vertices_.size() << "\n";
        std::cout << "  Ghost cells: " << ghost_cells_.size() << "\n";
    }

#endif
}

// Helper function to synchronize field data for ghost entities
void DistributedMesh::synchronize_field_data(EntityKind kind, const std::string& field_name) {
#ifdef MESH_HAS_MPI
    // =====================================================
    // Field Data Synchronization Algorithm
    // =====================================================
    // Purpose: Update ghost entity field values from owning ranks
    // Performance: O(n_ghosts * field_size) per field
    // Communication: Uses established exchange patterns for efficiency

    if (world_size_ == 1) return;

    // Get field information
    void* field_data = local_mesh_->field_data_by_name(kind, field_name);
    if (!field_data) return;  // Field doesn't exist

    size_t components = local_mesh_->field_components_by_name(kind, field_name);
    size_t bytes_per_comp = local_mesh_->field_bytes_per_component_by_name(kind, field_name);
    size_t bytes_per_entity = components * bytes_per_comp;

    // Use appropriate exchange pattern based on entity type
    const ExchangePattern* pattern = nullptr;
    if (kind == EntityKind::Vertex) {
        pattern = &vertex_exchange_;
    } else if (kind == EntityKind::Volume) {
        pattern = &cell_exchange_;
    } else if (kind == EntityKind::Face) {
        pattern = &face_exchange_;
    } else {
        return;  // Unsupported entity type
    }

    // Check if exchange pattern is built
    if (pattern->send_ranks.empty() && pattern->recv_ranks.empty()) {
        return;  // No communication needed
    }

    // Prepare send buffers
    std::vector<std::vector<char>> send_buffers(pattern->send_ranks.size());
    for (size_t i = 0; i < pattern->send_ranks.size(); ++i) {
        size_t n_send = pattern->send_lists[i].size();
        send_buffers[i].resize(n_send * bytes_per_entity);

        // Pack field data for entities to send
        char* send_ptr = send_buffers[i].data();
        for (size_t j = 0; j < n_send; ++j) {
            index_t entity_id = pattern->send_lists[i][j];
            const char* src = static_cast<const char*>(field_data) + entity_id * bytes_per_entity;
            std::memcpy(send_ptr + j * bytes_per_entity, src, bytes_per_entity);
        }
    }

    // Prepare receive buffers
    std::vector<std::vector<char>> recv_buffers(pattern->recv_ranks.size());
    for (size_t i = 0; i < pattern->recv_ranks.size(); ++i) {
        size_t n_recv = pattern->recv_lists[i].size();
        recv_buffers[i].resize(n_recv * bytes_per_entity);
    }

    // Non-blocking send/receive
    std::vector<MPI_Request> requests;
    requests.reserve(pattern->send_ranks.size() + pattern->recv_ranks.size());

    // Post receives first (for better performance)
    for (size_t i = 0; i < pattern->recv_ranks.size(); ++i) {
        if (!recv_buffers[i].empty()) {
            MPI_Request req;
            MPI_Irecv(recv_buffers[i].data(), static_cast<int>(recv_buffers[i].size()),
                     MPI_BYTE, pattern->recv_ranks[i], 300, comm_, &req);
            requests.push_back(req);
        }
    }

    // Post sends
    for (size_t i = 0; i < pattern->send_ranks.size(); ++i) {
        if (!send_buffers[i].empty()) {
            MPI_Request req;
            MPI_Isend(send_buffers[i].data(), static_cast<int>(send_buffers[i].size()),
                     MPI_BYTE, pattern->send_ranks[i], 300, comm_, &req);
            requests.push_back(req);
        }
    }

    // Wait for all communication to complete
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    // Unpack received field data
    for (size_t i = 0; i < pattern->recv_ranks.size(); ++i) {
        const char* recv_ptr = recv_buffers[i].data();
        for (size_t j = 0; j < pattern->recv_lists[i].size(); ++j) {
            index_t entity_id = pattern->recv_lists[i][j];
            char* dst = static_cast<char*>(field_data) + entity_id * bytes_per_entity;
            std::memcpy(dst, recv_ptr + j * bytes_per_entity, bytes_per_entity);
        }
    }
#endif
}

} // namespace svmp

#endif // MESH_HAS_MPI && !MESH_BUILD_TESTS
