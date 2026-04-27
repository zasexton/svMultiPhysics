/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/TopologyAnalysisContext.h"
#include "Assembly/Assembler.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <set>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

class DisjointCellSet {
public:
    explicit DisjointCellSet(GlobalIndex n_cells)
        : parent_(static_cast<std::size_t>(n_cells)),
          rank_(static_cast<std::size_t>(n_cells), 0u)
    {
        std::iota(parent_.begin(), parent_.end(), GlobalIndex{0});
    }

    [[nodiscard]] GlobalIndex find(GlobalIndex cell)
    {
        auto idx = static_cast<std::size_t>(cell);
        while (parent_[idx] != cell) {
            parent_[idx] = parent_[static_cast<std::size_t>(parent_[idx])];
            cell = parent_[idx];
            idx = static_cast<std::size_t>(cell);
        }
        return cell;
    }

    void unite(GlobalIndex a, GlobalIndex b)
    {
        auto root_a = find(a);
        auto root_b = find(b);
        if (root_a == root_b) {
            return;
        }

        auto idx_a = static_cast<std::size_t>(root_a);
        auto idx_b = static_cast<std::size_t>(root_b);
        if (rank_[idx_a] < rank_[idx_b]) {
            std::swap(root_a, root_b);
            std::swap(idx_a, idx_b);
        }

        parent_[idx_b] = root_a;
        if (rank_[idx_a] == rank_[idx_b]) {
            ++rank_[idx_a];
        }
    }

private:
    std::vector<GlobalIndex> parent_;
    std::vector<std::uint8_t> rank_;
};

} // namespace

// ============================================================================
// Queries
// ============================================================================

int TopologyAnalysisContext::regionForCell(GlobalIndex cell_idx) const noexcept {
    if (cell_idx < 0 || static_cast<std::size_t>(cell_idx) >= cell_to_region_.size()) {
        return -1;
    }
    return cell_to_region_[static_cast<std::size_t>(cell_idx)];
}

std::vector<int> TopologyAnalysisContext::regionsForBoundaryMarker(int marker) const {
    auto it = boundary_mapping.marker_to_regions.find(marker);
    if (it == boundary_mapping.marker_to_regions.end()) return {};
    return it->second;
}

// ============================================================================
// Factory: build from IMeshAccess
// ============================================================================

TopologyAnalysisContext TopologyAnalysisContext::build(const assembly::IMeshAccess& mesh) {
    TopologyAnalysisContext ctx;

    const auto n_cells = mesh.numCells();
    if (n_cells <= 0) return ctx;

    // --- Step 1: Union cells that share a node.  This avoids materializing the
    // full cell-cell adjacency graph, which is prohibitively expensive for large
    // tetrahedral meshes.
    DisjointCellSet components(n_cells);
    std::unordered_map<GlobalIndex, GlobalIndex> first_cell_for_node;
    first_cell_for_node.reserve(static_cast<std::size_t>(
        std::min<GlobalIndex>(std::max<GlobalIndex>(n_cells / 2, 1), 2000000)));
    std::vector<GlobalIndex> cell_nodes_buf;
    GlobalIndex max_node_id = -1;

    for (GlobalIndex c = 0; c < n_cells; ++c) {
        cell_nodes_buf.clear();
        mesh.getCellNodes(c, cell_nodes_buf);
        for (auto n : cell_nodes_buf) {
            if (n < 0) {
                continue;
            }
            max_node_id = std::max(max_node_id, n);
            auto [it, inserted] = first_cell_for_node.emplace(n, c);
            if (!inserted) {
                components.unite(c, it->second);
            }
        }
    }
    std::unordered_map<GlobalIndex, GlobalIndex>().swap(first_cell_for_node);

    // --- Step 2: Assign compact region IDs and collect cells per component. ---
    ctx.cell_to_region_.assign(static_cast<std::size_t>(n_cells), -1);
    std::unordered_map<GlobalIndex, int> root_to_region;
    root_to_region.reserve(static_cast<std::size_t>(std::min<GlobalIndex>(n_cells, 1024)));

    for (GlobalIndex c = 0; c < n_cells; ++c) {
        const auto root = components.find(c);
        auto it = root_to_region.find(root);
        if (it == root_to_region.end()) {
            const auto region_id = static_cast<int>(ctx.components.size());
            ConnectedComponent comp;
            comp.region_id = region_id;
            ctx.components.push_back(std::move(comp));
            it = root_to_region.emplace(root, region_id).first;
        }
        const int region_id = it->second;
        ctx.cell_to_region_[static_cast<std::size_t>(c)] = region_id;
        ctx.components[static_cast<std::size_t>(region_id)].cell_indices.push_back(c);
    }

    // Count unique vertices per component without storing per-component sets.
    std::vector<int> vertex_region_seen;
    std::unordered_map<GlobalIndex, int> sparse_vertex_region_seen;
    if (max_node_id >= 0 && static_cast<std::uint64_t>(max_node_id) <= 50000000ULL) {
        vertex_region_seen.assign(static_cast<std::size_t>(max_node_id) + 1u, -1);
    } else {
        sparse_vertex_region_seen.reserve(static_cast<std::size_t>(
            std::min<GlobalIndex>(std::max<GlobalIndex>(n_cells / 2, 1), 2000000)));
    }

    for (auto& comp : ctx.components) {
        comp.num_cells = static_cast<int>(comp.cell_indices.size());
        comp.num_vertices = 0;
    }

    for (GlobalIndex c = 0; c < n_cells; ++c) {
        const int region_id = ctx.cell_to_region_[static_cast<std::size_t>(c)];
        if (region_id < 0) {
            continue;
        }
        auto& comp = ctx.components[static_cast<std::size_t>(region_id)];
        cell_nodes_buf.clear();
        mesh.getCellNodes(c, cell_nodes_buf);
        for (auto n : cell_nodes_buf) {
            if (n < 0) {
                continue;
            }
            if (!vertex_region_seen.empty() &&
                static_cast<std::size_t>(n) < vertex_region_seen.size()) {
                auto& seen_region = vertex_region_seen[static_cast<std::size_t>(n)];
                if (seen_region < 0) {
                    seen_region = region_id;
                    ++comp.num_vertices;
                }
                continue;
            }
            auto [it, inserted] = sparse_vertex_region_seen.emplace(n, region_id);
            if (inserted) {
                ++comp.num_vertices;
            }
        }
    }

    // --- Step 3: Map boundary markers to regions ---
    std::set<int> all_markers;
    bool visited_boundary_face = false;
    mesh.forEachBoundaryFace(/*marker=*/-1, [&](GlobalIndex face_id, GlobalIndex /*cell_id*/) {
        visited_boundary_face = true;
        int marker = mesh.getBoundaryFaceMarker(face_id);
        if (marker >= 0) {
            all_markers.insert(marker);
        }
    });
    if (!visited_boundary_face) {
        const auto n_boundary = mesh.numBoundaryFaces();
        for (GlobalIndex f = 0; f < n_boundary; ++f) {
            int marker = mesh.getBoundaryFaceMarker(f);
            if (marker >= 0) {
                all_markers.insert(marker);
            }
        }
    }

    for (int marker : all_markers) {
        std::set<int> regions_for_marker;

        mesh.forEachBoundaryFace(marker, [&](GlobalIndex /*face_id*/, GlobalIndex cell_id) {
            int r = ctx.regionForCell(cell_id);
            if (r >= 0) {
                regions_for_marker.insert(r);
            }
        });

        std::vector<int> regions_vec(regions_for_marker.begin(), regions_for_marker.end());
        ctx.boundary_mapping.marker_to_regions[marker] = regions_vec;

        for (int r : regions_vec) {
            ctx.boundary_mapping.region_to_markers[r].insert(marker);
        }

        // Also annotate the ConnectedComponent objects
        for (int r : regions_vec) {
            if (r >= 0 && static_cast<std::size_t>(r) < ctx.components.size()) {
                ctx.components[static_cast<std::size_t>(r)].boundary_markers.insert(marker);
            }
        }
    }

    // Note: interface face detection was moved to InterfaceTopologyContext (Phase 14).
    // TopologyAnalysisContext is now for bulk connected components only.

    return ctx;
}

} // namespace analysis
} // namespace FE
} // namespace svmp
