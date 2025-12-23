/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "DofGraph.h"
#include <algorithm>
#include <unordered_set>
#include <numeric>
#include <cmath>

namespace svmp {
namespace FE {
namespace dofs {

// =============================================================================
// Construction
// =============================================================================

DofGraph::DofGraph() = default;
DofGraph::~DofGraph() = default;

DofGraph::DofGraph(DofGraph&&) noexcept = default;
DofGraph& DofGraph::operator=(DofGraph&&) noexcept = default;

// =============================================================================
// Graph Building
// =============================================================================

void DofGraph::build(const DofMap& dof_map, const DofGraphOptions& options) {
    switch (options.pattern) {
        case CouplingPattern::CellOnly:
        case CouplingPattern::BlockDiagonal:
            buildCellOnly(dof_map, options);
            break;

        case CouplingPattern::CellPlusFace:
        case CouplingPattern::DGCoupling:
            // For DG patterns without face info, fall back to cell-only
            buildCellOnly(dof_map, options);
            break;
    }

    if (options.symmetric) {
        symmetrize();
    }

    if (options.remove_duplicates) {
        removeDuplicates();
    }

    sortIndices();
    symmetric_ = options.symmetric;
}

void DofGraph::buildCellOnly(const DofMap& dof_map, const DofGraphOptions& options) {
    n_dofs_ = dof_map.getNumDofs();

    if (n_dofs_ == 0) {
        row_offsets_.assign(1, 0);
        col_indices_.clear();
        valid_ = true;
        return;
    }

    // Count edges per DOF (using hash sets for uniqueness)
    std::vector<std::unordered_set<GlobalIndex>> adjacency(static_cast<std::size_t>(n_dofs_));

    auto n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto cell_dofs = dof_map.getCellDofs(c);

        // All DOFs in a cell are connected
        for (auto dof_i : cell_dofs) {
            if (dof_i < 0 || dof_i >= n_dofs_) continue;

            auto& adj_set = adjacency[static_cast<std::size_t>(dof_i)];

            for (auto dof_j : cell_dofs) {
                if (dof_j < 0 || dof_j >= n_dofs_) continue;

                // Include diagonal?
                if (!options.include_diagonal && dof_i == dof_j) continue;

                adj_set.insert(dof_j);
            }
        }
    }

    // Convert to CSR
    row_offsets_.clear();
    row_offsets_.reserve(static_cast<std::size_t>(n_dofs_ + 1));
    col_indices_.clear();

    GlobalIndex offset = 0;
    for (GlobalIndex d = 0; d < n_dofs_; ++d) {
        row_offsets_.push_back(offset);
        const auto& adj_set = adjacency[static_cast<std::size_t>(d)];

        for (auto neighbor : adj_set) {
            col_indices_.push_back(neighbor);
        }

        offset += static_cast<GlobalIndex>(adj_set.size());
    }
    row_offsets_.push_back(offset);

    valid_ = true;
}

void DofGraph::buildWithFaces(const DofMap& dof_map,
                               std::span<const GlobalIndex> cell_neighbors,
                               std::span<const GlobalIndex> cell_neighbor_offsets,
                               const DofGraphOptions& options) {
    // First build cell-only connectivity
    buildCellOnly(dof_map, options);

    if (options.pattern == CouplingPattern::CellOnly) {
        return;  // No face coupling needed
    }

    // Add face coupling
    n_dofs_ = dof_map.getNumDofs();
    std::vector<std::unordered_set<GlobalIndex>> adjacency(static_cast<std::size_t>(n_dofs_));

    // Copy existing adjacency
    for (GlobalIndex d = 0; d < n_dofs_; ++d) {
        auto neighbors = getNeighbors(d);
        adjacency[static_cast<std::size_t>(d)].insert(neighbors.begin(), neighbors.end());
    }

    // Add face neighbors
    auto n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto cell_dofs = dof_map.getCellDofs(c);

        // Get neighbor cells
        auto neigh_start = static_cast<std::size_t>(cell_neighbor_offsets[static_cast<std::size_t>(c)]);
        auto neigh_end = static_cast<std::size_t>(cell_neighbor_offsets[static_cast<std::size_t>(c) + 1]);

        for (std::size_t n = neigh_start; n < neigh_end && n < cell_neighbors.size(); ++n) {
            GlobalIndex neighbor_cell = cell_neighbors[n];
            if (neighbor_cell < 0 || neighbor_cell >= n_cells) continue;

            auto neighbor_dofs = dof_map.getCellDofs(neighbor_cell);

            // Connect all DOFs between cells
            for (auto dof_i : cell_dofs) {
                if (dof_i < 0 || dof_i >= n_dofs_) continue;
                auto& adj_set = adjacency[static_cast<std::size_t>(dof_i)];

                for (auto dof_j : neighbor_dofs) {
                    if (dof_j < 0 || dof_j >= n_dofs_) continue;
                    adj_set.insert(dof_j);
                }
            }
        }
    }

    // Rebuild CSR
    row_offsets_.clear();
    row_offsets_.reserve(static_cast<std::size_t>(n_dofs_ + 1));
    col_indices_.clear();

    GlobalIndex offset = 0;
    for (GlobalIndex d = 0; d < n_dofs_; ++d) {
        row_offsets_.push_back(offset);
        const auto& adj_set = adjacency[static_cast<std::size_t>(d)];

        for (auto neighbor : adj_set) {
            col_indices_.push_back(neighbor);
        }

        offset += static_cast<GlobalIndex>(adj_set.size());
    }
    row_offsets_.push_back(offset);

    if (options.symmetric) {
        symmetrize();
    }

    sortIndices();
}

void DofGraph::buildFromCSR(GlobalIndex n_dofs,
                             std::span<const GlobalIndex> row_offsets,
                             std::span<const GlobalIndex> col_indices) {
    n_dofs_ = n_dofs;
    row_offsets_.assign(row_offsets.begin(), row_offsets.end());
    col_indices_.assign(col_indices.begin(), col_indices.end());
    valid_ = true;
    symmetric_ = false;  // Unknown
}

void DofGraph::invalidate() {
    valid_ = false;
}

// =============================================================================
// Graph Access
// =============================================================================

std::span<const GlobalIndex> DofGraph::getNeighbors(GlobalIndex dof) const {
    if (!valid_ || dof < 0 || dof >= n_dofs_) {
        return {};
    }

    auto start = static_cast<std::size_t>(row_offsets_[static_cast<std::size_t>(dof)]);
    auto end = static_cast<std::size_t>(row_offsets_[static_cast<std::size_t>(dof) + 1]);

    if (end > col_indices_.size()) {
        end = col_indices_.size();
    }

    return {col_indices_.data() + start, end - start};
}

GlobalIndex DofGraph::getDegree(GlobalIndex dof) const {
    if (!valid_ || dof < 0 || dof >= n_dofs_) {
        return 0;
    }

    auto start = row_offsets_[static_cast<std::size_t>(dof)];
    auto end = row_offsets_[static_cast<std::size_t>(dof) + 1];

    return end - start;
}

// =============================================================================
// Graph Statistics
// =============================================================================

GlobalIndex DofGraph::getBandwidth() const {
    if (!valid_) return 0;

    GlobalIndex bandwidth = 0;

    for (GlobalIndex d = 0; d < n_dofs_; ++d) {
        auto neighbors = getNeighbors(d);
        for (auto n : neighbors) {
            GlobalIndex diff = std::abs(d - n);
            bandwidth = std::max(bandwidth, diff);
        }
    }

    return bandwidth;
}

GlobalIndex DofGraph::getMaxRowNnz() const {
    if (!valid_) return 0;

    GlobalIndex max_nnz = 0;
    for (GlobalIndex d = 0; d < n_dofs_; ++d) {
        max_nnz = std::max(max_nnz, getDegree(d));
    }

    return max_nnz;
}

double DofGraph::getAvgRowNnz() const {
    if (!valid_ || n_dofs_ == 0) return 0.0;

    return static_cast<double>(col_indices_.size()) / n_dofs_;
}

bool DofGraph::isSymmetric() const {
    if (!valid_) return false;

    // Check if for every edge (i,j) there is also (j,i)
    for (GlobalIndex d = 0; d < n_dofs_; ++d) {
        auto neighbors = getNeighbors(d);
        for (auto n : neighbors) {
            if (n < 0 || n >= n_dofs_) continue;

            // Check if d is in neighbors of n
            auto reverse_neighbors = getNeighbors(n);
            bool found = false;
            for (auto rn : reverse_neighbors) {
                if (rn == d) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
    }

    return true;
}

GlobalIndex DofGraph::getProfile() const {
    if (!valid_) return 0;

    GlobalIndex profile = 0;

    for (GlobalIndex d = 0; d < n_dofs_; ++d) {
        auto neighbors = getNeighbors(d);
        if (neighbors.empty()) continue;

        GlobalIndex min_col = d;
        for (auto n : neighbors) {
            min_col = std::min(min_col, n);
        }

        profile += (d - min_col);
    }

    return profile;
}

DofGraph::Statistics DofGraph::getStatistics() const {
    Statistics stats;

    if (!valid_) return stats;

    stats.n_dofs = n_dofs_;
    stats.n_edges = static_cast<GlobalIndex>(col_indices_.size());
    stats.bandwidth = getBandwidth();
    stats.max_degree = getMaxRowNnz();
    stats.avg_degree = getAvgRowNnz();
    stats.profile = getProfile();
    stats.symmetric = symmetric_ || isSymmetric();

    return stats;
}

// =============================================================================
// Graph Manipulation
// =============================================================================

void DofGraph::symmetrize() {
    if (!valid_ || n_dofs_ == 0) return;

    std::vector<std::unordered_set<GlobalIndex>> adjacency(static_cast<std::size_t>(n_dofs_));

    // Copy existing edges
    for (GlobalIndex d = 0; d < n_dofs_; ++d) {
        auto neighbors = getNeighbors(d);
        adjacency[static_cast<std::size_t>(d)].insert(neighbors.begin(), neighbors.end());
    }

    // Add reverse edges
    for (GlobalIndex d = 0; d < n_dofs_; ++d) {
        for (auto n : adjacency[static_cast<std::size_t>(d)]) {
            if (n >= 0 && n < n_dofs_) {
                adjacency[static_cast<std::size_t>(n)].insert(d);
            }
        }
    }

    // Rebuild CSR
    row_offsets_.clear();
    col_indices_.clear();

    GlobalIndex offset = 0;
    for (GlobalIndex d = 0; d < n_dofs_; ++d) {
        row_offsets_.push_back(offset);
        const auto& adj = adjacency[static_cast<std::size_t>(d)];
        col_indices_.insert(col_indices_.end(), adj.begin(), adj.end());
        offset += static_cast<GlobalIndex>(adj.size());
    }
    row_offsets_.push_back(offset);

    symmetric_ = true;
}

void DofGraph::removeDuplicates() {
    if (!valid_) return;

    // Sort and unique within each row
    for (GlobalIndex d = 0; d < n_dofs_; ++d) {
        auto start = static_cast<std::size_t>(row_offsets_[static_cast<std::size_t>(d)]);
        auto end = static_cast<std::size_t>(row_offsets_[static_cast<std::size_t>(d) + 1]);

        if (start < end && end <= col_indices_.size()) {
            std::sort(col_indices_.begin() + start, col_indices_.begin() + end);
        }
    }

    // Rebuild without duplicates
    std::vector<GlobalIndex> new_col_indices;
    std::vector<GlobalIndex> new_row_offsets;
    new_row_offsets.reserve(row_offsets_.size());

    GlobalIndex offset = 0;
    for (GlobalIndex d = 0; d < n_dofs_; ++d) {
        new_row_offsets.push_back(offset);

        auto start = static_cast<std::size_t>(row_offsets_[static_cast<std::size_t>(d)]);
        auto end = static_cast<std::size_t>(row_offsets_[static_cast<std::size_t>(d) + 1]);

        GlobalIndex last = -1;
        for (std::size_t i = start; i < end && i < col_indices_.size(); ++i) {
            if (col_indices_[i] != last) {
                new_col_indices.push_back(col_indices_[i]);
                last = col_indices_[i];
                ++offset;
            }
        }
    }
    new_row_offsets.push_back(offset);

    row_offsets_ = std::move(new_row_offsets);
    col_indices_ = std::move(new_col_indices);
}

void DofGraph::sortIndices() {
    if (!valid_) return;

    for (GlobalIndex d = 0; d < n_dofs_; ++d) {
        auto start = static_cast<std::size_t>(row_offsets_[static_cast<std::size_t>(d)]);
        auto end = static_cast<std::size_t>(row_offsets_[static_cast<std::size_t>(d) + 1]);

        if (start < end && end <= col_indices_.size()) {
            std::sort(col_indices_.begin() + start, col_indices_.begin() + end);
        }
    }
}

DofGraph DofGraph::applyPermutation(std::span<const GlobalIndex> permutation) const {
    DofGraph result;

    if (!valid_ || permutation.size() != static_cast<std::size_t>(n_dofs_)) {
        return result;
    }

    // Create inverse permutation
    std::vector<GlobalIndex> inverse(static_cast<std::size_t>(n_dofs_));
    for (std::size_t i = 0; i < permutation.size(); ++i) {
        inverse[static_cast<std::size_t>(permutation[i])] = static_cast<GlobalIndex>(i);
    }

    // Build new adjacency
    std::vector<std::vector<GlobalIndex>> new_adj(static_cast<std::size_t>(n_dofs_));

    for (GlobalIndex old_d = 0; old_d < n_dofs_; ++old_d) {
        GlobalIndex new_d = permutation[static_cast<std::size_t>(old_d)];
        auto neighbors = getNeighbors(old_d);

        for (auto old_n : neighbors) {
            if (old_n >= 0 && static_cast<std::size_t>(old_n) < permutation.size()) {
                GlobalIndex new_n = permutation[static_cast<std::size_t>(old_n)];
                new_adj[static_cast<std::size_t>(new_d)].push_back(new_n);
            }
        }
    }

    // Build CSR
    result.n_dofs_ = n_dofs_;
    result.row_offsets_.clear();
    result.col_indices_.clear();

    GlobalIndex offset = 0;
    for (GlobalIndex d = 0; d < n_dofs_; ++d) {
        result.row_offsets_.push_back(offset);
        auto& adj = new_adj[static_cast<std::size_t>(d)];
        std::sort(adj.begin(), adj.end());
        result.col_indices_.insert(result.col_indices_.end(), adj.begin(), adj.end());
        offset += static_cast<GlobalIndex>(adj.size());
    }
    result.row_offsets_.push_back(offset);

    result.valid_ = true;
    result.symmetric_ = symmetric_;

    return result;
}

DofGraph DofGraph::extractSubgraph(std::span<const GlobalIndex> dof_subset) const {
    DofGraph result;

    if (!valid_ || dof_subset.empty()) {
        return result;
    }

    // Create mapping from old to new indices
    std::unordered_map<GlobalIndex, GlobalIndex> old_to_new;
    for (std::size_t i = 0; i < dof_subset.size(); ++i) {
        old_to_new[dof_subset[i]] = static_cast<GlobalIndex>(i);
    }

    result.n_dofs_ = static_cast<GlobalIndex>(dof_subset.size());
    result.row_offsets_.clear();
    result.col_indices_.clear();

    GlobalIndex offset = 0;
    for (auto old_d : dof_subset) {
        result.row_offsets_.push_back(offset);

        auto neighbors = getNeighbors(old_d);
        for (auto old_n : neighbors) {
            auto it = old_to_new.find(old_n);
            if (it != old_to_new.end()) {
                result.col_indices_.push_back(it->second);
                ++offset;
            }
        }
    }
    result.row_offsets_.push_back(offset);

    result.valid_ = true;
    result.symmetric_ = symmetric_;

    return result;
}

// =============================================================================
// Utility Functions
// =============================================================================

void computeSparsityPattern(const DofGraph& graph,
                            std::vector<GlobalIndex>& row_offsets,
                            std::vector<GlobalIndex>& col_indices) {
    row_offsets.assign(graph.getAdjOffsets().begin(), graph.getAdjOffsets().end());
    col_indices.assign(graph.getAdjIndices().begin(), graph.getAdjIndices().end());
}

GlobalIndex estimateFillIn(const DofGraph& graph) {
    // Simple fill-in estimation using symbolic Cholesky
    // This is a rough estimate

    if (!graph.isValid()) return 0;

    auto n = graph.numDofs();
    auto existing_nnz = graph.numEdges();

    // Use a simple heuristic: fill-in is roughly proportional to
    // bandwidth * average_degree
    GlobalIndex bandwidth = graph.getBandwidth();
    double avg_degree = graph.getAvgRowNnz();

    // Rough estimate
    auto estimated_total = static_cast<GlobalIndex>(n * avg_degree * std::log(bandwidth + 1));

    return std::max(GlobalIndex{0}, estimated_total - existing_nnz);
}

} // namespace dofs
} // namespace FE
} // namespace svmp
