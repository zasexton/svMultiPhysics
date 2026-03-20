/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/TopologyAnalysisContext.h"
#include "Assembly/Assembler.h"

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace analysis {

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

    // --- Step 1: Build cell-cell adjacency via shared nodes ---
    // For each node, collect the cells that reference it.
    // Two cells sharing a node are neighbors.

    std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> node_to_cells;
    std::vector<GlobalIndex> cell_nodes_buf;

    for (GlobalIndex c = 0; c < n_cells; ++c) {
        cell_nodes_buf.clear();
        mesh.getCellNodes(c, cell_nodes_buf);
        for (auto n : cell_nodes_buf) {
            node_to_cells[n].push_back(c);
        }
    }

    // Build adjacency list (symmetric, deduplicated)
    std::vector<std::unordered_set<GlobalIndex>> adj(static_cast<std::size_t>(n_cells));
    for (const auto& [node, cells] : node_to_cells) {
        for (std::size_t i = 0; i < cells.size(); ++i) {
            for (std::size_t j = i + 1; j < cells.size(); ++j) {
                adj[static_cast<std::size_t>(cells[i])].insert(cells[j]);
                adj[static_cast<std::size_t>(cells[j])].insert(cells[i]);
            }
        }
    }

    // --- Step 2: BFS for connected components ---
    ctx.cell_to_region_.assign(static_cast<std::size_t>(n_cells), -1);
    int region_id = 0;

    for (GlobalIndex start = 0; start < n_cells; ++start) {
        if (ctx.cell_to_region_[static_cast<std::size_t>(start)] >= 0) continue;

        ConnectedComponent comp;
        comp.region_id = region_id;

        std::queue<GlobalIndex> queue;
        queue.push(start);
        ctx.cell_to_region_[static_cast<std::size_t>(start)] = region_id;

        // Count unique vertices in this component
        std::unordered_set<GlobalIndex> comp_vertices;

        while (!queue.empty()) {
            GlobalIndex c = queue.front();
            queue.pop();
            comp.cell_indices.push_back(c);

            // Collect vertices
            cell_nodes_buf.clear();
            mesh.getCellNodes(c, cell_nodes_buf);
            for (auto n : cell_nodes_buf) {
                comp_vertices.insert(n);
            }

            for (auto neighbor : adj[static_cast<std::size_t>(c)]) {
                auto& r = ctx.cell_to_region_[static_cast<std::size_t>(neighbor)];
                if (r < 0) {
                    r = region_id;
                    queue.push(neighbor);
                }
            }
        }

        comp.num_cells = static_cast<int>(comp.cell_indices.size());
        comp.num_vertices = static_cast<int>(comp_vertices.size());
        ctx.components.push_back(std::move(comp));
        ++region_id;
    }

    // --- Step 3: Map boundary markers to regions ---
    const auto n_boundary = mesh.numBoundaryFaces();

    // We need to iterate over all boundary faces. IMeshAccess provides
    // forEachBoundaryFace(marker, callback), but we need all markers.
    // Iterate over all boundary face IDs and query their markers.
    std::unordered_set<int> all_markers;
    for (GlobalIndex f = 0; f < n_boundary; ++f) {
        int marker = mesh.getBoundaryFaceMarker(f);
        if (marker >= 0) {
            all_markers.insert(marker);
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
