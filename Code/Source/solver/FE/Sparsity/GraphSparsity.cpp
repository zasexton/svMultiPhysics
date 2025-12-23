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

#include "GraphSparsity.h"
#include "SparsityOps.h"
#include <algorithm>
#include <queue>
#include <set>
#include <limits>
#include <numeric>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// ColoringResult Implementation
// ============================================================================

std::vector<GlobalIndex> ColoringResult::getVerticesOfColor(GlobalIndex color) const {
    std::vector<GlobalIndex> vertices;
    if (color < 0 || color >= num_colors) return vertices;

    vertices.reserve(static_cast<std::size_t>(color_counts[static_cast<std::size_t>(color)]));
    for (std::size_t v = 0; v < colors.size(); ++v) {
        if (colors[v] == color) {
            vertices.push_back(static_cast<GlobalIndex>(v));
        }
    }
    return vertices;
}

// ============================================================================
// ComponentResult Implementation
// ============================================================================

std::vector<GlobalIndex> ComponentResult::getVerticesInComponent(GlobalIndex comp) const {
    std::vector<GlobalIndex> vertices;
    if (comp < 0 || comp >= num_components) return vertices;

    vertices.reserve(static_cast<std::size_t>(component_sizes[static_cast<std::size_t>(comp)]));
    for (std::size_t v = 0; v < component_id.size(); ++v) {
        if (component_id[v] == comp) {
            vertices.push_back(static_cast<GlobalIndex>(v));
        }
    }
    return vertices;
}

// ============================================================================
// GraphSparsity Construction
// ============================================================================

GraphSparsity::GraphSparsity(const SparsityPattern& pattern, bool symmetrize)
    : pattern_(pattern), is_symmetric_(symmetrize)
{
    if (!pattern_.isFinalized()) {
        pattern_.finalize();
    }
    if (symmetrize) {
        ensureSymmetric();
    }
}

GraphSparsity::GraphSparsity(SparsityPattern&& pattern, bool symmetrize)
    : pattern_(std::move(pattern)), is_symmetric_(symmetrize)
{
    if (!pattern_.isFinalized()) {
        pattern_.finalize();
    }
    if (symmetrize) {
        ensureSymmetric();
    }
}

void GraphSparsity::setPattern(const SparsityPattern& pattern, bool symmetrize) {
    pattern_ = pattern;
    is_symmetric_ = symmetrize;
    if (!pattern_.isFinalized()) {
        pattern_.finalize();
    }
    if (symmetrize) {
        ensureSymmetric();
    }
}

void GraphSparsity::ensureSymmetric() {
    if (!pattern_.isSquare()) {
        FE_THROW(InvalidArgumentException, "Graph operations require square pattern");
    }

    if (!pattern_.isSymmetric()) {
        pattern_ = symmetrizePattern(pattern_);
    }

    if (!pattern_.isFinalized()) {
        pattern_.finalize();
    }
}

// ============================================================================
// Graph Statistics
// ============================================================================

GraphStats GraphSparsity::computeStats() const {
    GraphStats stats;
    stats.n_vertices = pattern_.numRows();

    if (stats.n_vertices == 0) return stats;

    // Compute degrees
    auto degrees = computeDegrees();

    stats.min_degree = *std::min_element(degrees.begin(), degrees.end());
    stats.max_degree = *std::max_element(degrees.begin(), degrees.end());
    stats.avg_degree = static_cast<double>(
        std::accumulate(degrees.begin(), degrees.end(), GlobalIndex{0})) /
        static_cast<double>(stats.n_vertices);

    // Edge count (for symmetric graph, each edge counted twice in NNZ)
    stats.n_edges = pattern_.getNnz() / 2;

    // Bandwidth and profile
    stats.bandwidth = computeBandwidth();
    stats.profile = computeProfile();

    // Envelope is similar to profile
    stats.envelope = stats.profile;

    // Connected components
    auto components = computeConnectedComponents();
    stats.n_components = components.num_components;
    stats.is_connected = (stats.n_components == 1);

    return stats;
}

GlobalIndex GraphSparsity::computeBandwidth() const {
    return pattern_.computeBandwidth();
}

GlobalIndex GraphSparsity::computeProfile() const {
    GlobalIndex profile = 0;

    for (GlobalIndex row = 0; row < pattern_.numRows(); ++row) {
        auto row_span = pattern_.getRowSpan(row);
        if (!row_span.empty()) {
            GlobalIndex first_col = row_span[0];
            GlobalIndex last_col = row_span[row_span.size() - 1];
            profile += (last_col - first_col + 1);
        }
    }

    return profile;
}

std::vector<GlobalIndex> GraphSparsity::computeDegrees() const {
    std::vector<GlobalIndex> degrees(static_cast<std::size_t>(pattern_.numRows()));

    for (GlobalIndex row = 0; row < pattern_.numRows(); ++row) {
        // Degree is NNZ in row minus diagonal (self-loop)
        GlobalIndex nnz = pattern_.getRowNnz(row);
        if (pattern_.hasEntry(row, row)) {
            --nnz;
        }
        degrees[static_cast<std::size_t>(row)] = nnz;
    }

    return degrees;
}

GlobalIndex GraphSparsity::getDegree(GlobalIndex vertex) const {
    FE_CHECK_INDEX(vertex, pattern_.numRows());

    GlobalIndex nnz = pattern_.getRowNnz(vertex);
    if (pattern_.hasEntry(vertex, vertex)) {
        --nnz;
    }
    return nnz;
}

std::vector<GlobalIndex> GraphSparsity::getNeighbors(GlobalIndex vertex) const {
    FE_CHECK_INDEX(vertex, pattern_.numRows());

    std::vector<GlobalIndex> neighbors;
    auto row_span = pattern_.getRowSpan(vertex);

    neighbors.reserve(row_span.size());
    for (GlobalIndex col : row_span) {
        if (col != vertex) {
            neighbors.push_back(col);
        }
    }

    return neighbors;
}

// ============================================================================
// Graph Coloring
// ============================================================================

ColoringResult GraphSparsity::greedyColoring(std::span<const GlobalIndex> ordering) const {
    ColoringResult result;
    GlobalIndex n = pattern_.numRows();

    if (n == 0) {
        result.is_valid = true;
        return result;
    }

    result.colors.assign(static_cast<std::size_t>(n), -1);

    // Use provided ordering or natural order
    std::vector<GlobalIndex> order;
    if (ordering.empty()) {
        order.resize(static_cast<std::size_t>(n));
        std::iota(order.begin(), order.end(), GlobalIndex{0});
    } else {
        order.assign(ordering.begin(), ordering.end());
    }

    GlobalIndex max_color = 0;

    for (GlobalIndex v : order) {
        // Find colors used by neighbors
        std::set<GlobalIndex> neighbor_colors;
        for (GlobalIndex neighbor : getNeighbors(v)) {
            GlobalIndex c = result.colors[static_cast<std::size_t>(neighbor)];
            if (c >= 0) {
                neighbor_colors.insert(c);
            }
        }

        // Find smallest available color
        GlobalIndex color = 0;
        while (neighbor_colors.count(color) > 0) {
            ++color;
        }

        result.colors[static_cast<std::size_t>(v)] = color;
        max_color = std::max(max_color, color);
    }

    result.num_colors = max_color + 1;

    // Count vertices per color
    result.color_counts.assign(static_cast<std::size_t>(result.num_colors), 0);
    for (GlobalIndex c : result.colors) {
        ++result.color_counts[static_cast<std::size_t>(c)];
    }

    result.is_valid = verifyColoring(result.colors);
    return result;
}

ColoringResult GraphSparsity::degreeBasedColoring(bool use_smallest_last) const {
    GlobalIndex n = pattern_.numRows();

    if (n == 0) {
        ColoringResult result;
        result.is_valid = true;
        return result;
    }

    auto degrees = computeDegrees();

    // Create ordering based on degrees
    std::vector<GlobalIndex> ordering(static_cast<std::size_t>(n));
    std::iota(ordering.begin(), ordering.end(), GlobalIndex{0});

    if (use_smallest_last) {
        // Smallest degree last (SDL) - good for sparse graphs
        // Actually implemented as largest-first which is reversed SDL
        std::sort(ordering.begin(), ordering.end(),
            [&degrees](GlobalIndex a, GlobalIndex b) {
                return degrees[static_cast<std::size_t>(a)] >
                       degrees[static_cast<std::size_t>(b)];
            });
    } else {
        // Largest degree first (LDF)
        std::sort(ordering.begin(), ordering.end(),
            [&degrees](GlobalIndex a, GlobalIndex b) {
                return degrees[static_cast<std::size_t>(a)] >
                       degrees[static_cast<std::size_t>(b)];
            });
    }

    return greedyColoring(ordering);
}

bool GraphSparsity::verifyColoring(std::span<const GlobalIndex> colors) const {
    GlobalIndex n = pattern_.numRows();

    if (static_cast<GlobalIndex>(colors.size()) != n) {
        return false;
    }

    for (GlobalIndex v = 0; v < n; ++v) {
        GlobalIndex v_color = colors[static_cast<std::size_t>(v)];
        for (GlobalIndex neighbor : getNeighbors(v)) {
            if (colors[static_cast<std::size_t>(neighbor)] == v_color) {
                return false;
            }
        }
    }

    return true;
}

// ============================================================================
// Reordering Algorithms
// ============================================================================

std::vector<GlobalIndex> GraphSparsity::cuthillMcKee(GlobalIndex start_vertex) const {
    GlobalIndex n = pattern_.numRows();

    if (n == 0) return {};

    // Find start vertex if not specified
    if (start_vertex < 0 || start_vertex >= n) {
        start_vertex = findPseudoPeripheral();
    }

    std::vector<GlobalIndex> perm;
    perm.reserve(static_cast<std::size_t>(n));

    std::vector<bool> visited(static_cast<std::size_t>(n), false);
    auto degrees = computeDegrees();

    // BFS with degree-based ordering within levels
    std::queue<GlobalIndex> queue;
    queue.push(start_vertex);
    visited[static_cast<std::size_t>(start_vertex)] = true;

    while (!queue.empty() || perm.size() < static_cast<std::size_t>(n)) {
        if (queue.empty()) {
            // Find unvisited vertex with minimum degree (disconnected component)
            GlobalIndex min_degree_vertex = -1;
            GlobalIndex min_degree = std::numeric_limits<GlobalIndex>::max();
            for (GlobalIndex v = 0; v < n; ++v) {
                if (!visited[static_cast<std::size_t>(v)]) {
                    if (degrees[static_cast<std::size_t>(v)] < min_degree) {
                        min_degree = degrees[static_cast<std::size_t>(v)];
                        min_degree_vertex = v;
                    }
                }
            }
            if (min_degree_vertex >= 0) {
                queue.push(min_degree_vertex);
                visited[static_cast<std::size_t>(min_degree_vertex)] = true;
            } else {
                break;
            }
        }

        GlobalIndex v = queue.front();
        queue.pop();
        perm.push_back(v);

        // Get unvisited neighbors, sorted by degree
        std::vector<GlobalIndex> neighbors;
        for (GlobalIndex neighbor : getNeighbors(v)) {
            if (!visited[static_cast<std::size_t>(neighbor)]) {
                neighbors.push_back(neighbor);
                visited[static_cast<std::size_t>(neighbor)] = true;
            }
        }

        // Sort by degree (ascending)
        std::sort(neighbors.begin(), neighbors.end(),
            [&degrees](GlobalIndex a, GlobalIndex b) {
                return degrees[static_cast<std::size_t>(a)] <
                       degrees[static_cast<std::size_t>(b)];
            });

        for (GlobalIndex neighbor : neighbors) {
            queue.push(neighbor);
        }
    }

    return perm;
}

std::vector<GlobalIndex> GraphSparsity::reverseCuthillMcKee(GlobalIndex start_vertex) const {
    auto perm = cuthillMcKee(start_vertex);
    std::reverse(perm.begin(), perm.end());
    return perm;
}

GlobalIndex GraphSparsity::findPseudoPeripheral(GlobalIndex start) const {
    GlobalIndex n = pattern_.numRows();

    if (n == 0) return -1;
    if (start < 0 || start >= n) start = 0;

    GlobalIndex current = start;
    GlobalIndex current_ecc = computeEccentricity(current);

    // Iterate until eccentricity doesn't increase
    while (true) {
        // Find vertex at maximum distance from current
        auto distances = bfsDistances(current);

        GlobalIndex farthest = current;
        GlobalIndex max_dist = 0;
        auto degrees = computeDegrees();

        for (GlobalIndex v = 0; v < n; ++v) {
            GlobalIndex d = distances[static_cast<std::size_t>(v)];
            if (d > max_dist ||
                (d == max_dist &&
                 degrees[static_cast<std::size_t>(v)] <
                 degrees[static_cast<std::size_t>(farthest)])) {
                max_dist = d;
                farthest = v;
            }
        }

        if (max_dist <= current_ecc) {
            break;
        }

        current = farthest;
        current_ecc = max_dist;
    }

    return current;
}

std::vector<GlobalIndex> GraphSparsity::approximateMinimumDegree() const {
    GlobalIndex n = pattern_.numRows();

    if (n == 0) return {};

    // Copy adjacency for elimination
    std::vector<std::set<GlobalIndex>> adj(static_cast<std::size_t>(n));
    for (GlobalIndex row = 0; row < n; ++row) {
        for (GlobalIndex col : pattern_.getRowSpan(row)) {
            if (col != row) {
                adj[static_cast<std::size_t>(row)].insert(col);
            }
        }
    }

    std::vector<GlobalIndex> perm;
    perm.reserve(static_cast<std::size_t>(n));

    std::vector<bool> eliminated(static_cast<std::size_t>(n), false);

    // Priority queue: (degree, vertex)
    using PQEntry = std::pair<GlobalIndex, GlobalIndex>;
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;

    // Initialize priority queue
    for (GlobalIndex v = 0; v < n; ++v) {
        pq.emplace(static_cast<GlobalIndex>(adj[static_cast<std::size_t>(v)].size()), v);
    }

    while (perm.size() < static_cast<std::size_t>(n)) {
        // Find minimum degree vertex
        while (!pq.empty()) {
            auto [deg, v] = pq.top();
            pq.pop();

            if (eliminated[static_cast<std::size_t>(v)]) continue;

            // Check if degree is still accurate
            GlobalIndex current_deg = static_cast<GlobalIndex>(adj[static_cast<std::size_t>(v)].size());
            if (current_deg != deg) {
                // Re-insert with correct degree
                pq.emplace(current_deg, v);
                continue;
            }

            // Eliminate vertex
            perm.push_back(v);
            eliminated[static_cast<std::size_t>(v)] = true;

            // Update adjacency: connect all neighbors to each other
            std::vector<GlobalIndex> neighbors(adj[static_cast<std::size_t>(v)].begin(),
                                               adj[static_cast<std::size_t>(v)].end());

            for (std::size_t i = 0; i < neighbors.size(); ++i) {
                GlobalIndex u = neighbors[i];
                if (eliminated[static_cast<std::size_t>(u)]) continue;

                // Remove v from u's adjacency
                adj[static_cast<std::size_t>(u)].erase(v);

                // Add edges to other neighbors of v
                for (std::size_t j = i + 1; j < neighbors.size(); ++j) {
                    GlobalIndex w = neighbors[j];
                    if (!eliminated[static_cast<std::size_t>(w)]) {
                        adj[static_cast<std::size_t>(u)].insert(w);
                        adj[static_cast<std::size_t>(w)].insert(u);
                    }
                }

                // Re-add to priority queue with new degree
                pq.emplace(static_cast<GlobalIndex>(adj[static_cast<std::size_t>(u)].size()), u);
            }

            break;
        }
    }

    return perm;
}

std::vector<GlobalIndex> GraphSparsity::naturalOrdering() const {
    std::vector<GlobalIndex> perm(static_cast<std::size_t>(pattern_.numRows()));
    std::iota(perm.begin(), perm.end(), GlobalIndex{0});
    return perm;
}

// ============================================================================
// Structural Analysis
// ============================================================================

ComponentResult GraphSparsity::computeConnectedComponents() const {
    ComponentResult result;
    GlobalIndex n = pattern_.numRows();

    if (n == 0) return result;

    result.component_id.assign(static_cast<std::size_t>(n), -1);
    GlobalIndex current_component = 0;

    for (GlobalIndex start = 0; start < n; ++start) {
        if (result.component_id[static_cast<std::size_t>(start)] >= 0) continue;

        // BFS from this vertex
        std::queue<GlobalIndex> queue;
        queue.push(start);
        result.component_id[static_cast<std::size_t>(start)] = current_component;
        GlobalIndex component_size = 0;

        while (!queue.empty()) {
            GlobalIndex v = queue.front();
            queue.pop();
            ++component_size;

            for (GlobalIndex neighbor : getNeighbors(v)) {
                if (result.component_id[static_cast<std::size_t>(neighbor)] < 0) {
                    result.component_id[static_cast<std::size_t>(neighbor)] = current_component;
                    queue.push(neighbor);
                }
            }
        }

        result.component_sizes.push_back(component_size);
        ++current_component;
    }

    result.num_components = current_component;
    return result;
}

bool GraphSparsity::isConnected() const {
    auto components = computeConnectedComponents();
    return components.num_components <= 1;
}

LevelSetResult GraphSparsity::computeLevelSets(GlobalIndex root) const {
    LevelSetResult result;
    GlobalIndex n = pattern_.numRows();

    FE_CHECK_INDEX(root, n);

    result.root = root;
    result.levels.assign(static_cast<std::size_t>(n), -1);

    std::queue<GlobalIndex> queue;
    queue.push(root);
    result.levels[static_cast<std::size_t>(root)] = 0;

    GlobalIndex max_level = 0;

    while (!queue.empty()) {
        GlobalIndex v = queue.front();
        queue.pop();
        GlobalIndex v_level = result.levels[static_cast<std::size_t>(v)];

        for (GlobalIndex neighbor : getNeighbors(v)) {
            if (result.levels[static_cast<std::size_t>(neighbor)] < 0) {
                result.levels[static_cast<std::size_t>(neighbor)] = v_level + 1;
                max_level = std::max(max_level, v_level + 1);
                queue.push(neighbor);
            }
        }
    }

    result.num_levels = max_level + 1;

    // Build level sets
    result.level_sets.resize(static_cast<std::size_t>(result.num_levels));
    for (GlobalIndex v = 0; v < n; ++v) {
        GlobalIndex level = result.levels[static_cast<std::size_t>(v)];
        if (level >= 0) {
            result.level_sets[static_cast<std::size_t>(level)].push_back(v);
        }
    }

    return result;
}

GlobalIndex GraphSparsity::computeDiameter() const {
    GlobalIndex n = pattern_.numRows();
    if (n == 0) return 0;

    if (!isConnected()) return -1;

    GlobalIndex diameter = 0;

    // Sample vertices for diameter estimation (full computation is expensive)
    std::vector<GlobalIndex> sample_vertices;
    if (n <= 100) {
        sample_vertices.resize(static_cast<std::size_t>(n));
        std::iota(sample_vertices.begin(), sample_vertices.end(), GlobalIndex{0});
    } else {
        // Sample peripheral vertices
        GlobalIndex peripheral = findPseudoPeripheral();
        sample_vertices.push_back(peripheral);
        sample_vertices.push_back(0);
        sample_vertices.push_back(n - 1);
        sample_vertices.push_back(n / 2);
    }

    for (GlobalIndex v : sample_vertices) {
        GlobalIndex ecc = computeEccentricity(v);
        diameter = std::max(diameter, ecc);
    }

    return diameter;
}

GlobalIndex GraphSparsity::computeEccentricity(GlobalIndex vertex) const {
    auto distances = bfsDistances(vertex);
    GlobalIndex max_dist = 0;

    for (GlobalIndex d : distances) {
        if (d >= 0) {
            max_dist = std::max(max_dist, d);
        }
    }

    return max_dist;
}

// ============================================================================
// Fill-in Analysis
// ============================================================================

FillInPrediction GraphSparsity::predictCholeskyFillIn() const {
    FillInPrediction result;
    GlobalIndex n = pattern_.numRows();

    result.original_nnz = pattern_.getNnz();

    if (n == 0) {
        return result;
    }

    // Symbolic Cholesky factorization
    auto L_pattern = symbolicCholesky();
    result.total_factor_nnz = L_pattern.getNnz();
    result.predicted_fill = result.total_factor_nnz - result.original_nnz / 2;

    if (result.original_nnz > 0) {
        result.fill_ratio = static_cast<double>(result.total_factor_nnz) /
                           (static_cast<double>(result.original_nnz) / 2.0);
    }

    // Per-row fill-in
    result.row_fill.resize(static_cast<std::size_t>(n));
    for (GlobalIndex row = 0; row < n; ++row) {
        GlobalIndex original_row_nnz = 0;
        for (GlobalIndex col : pattern_.getRowSpan(row)) {
            if (col <= row) ++original_row_nnz;
        }
        result.row_fill[static_cast<std::size_t>(row)] =
            L_pattern.getRowNnz(row) - original_row_nnz;
    }

    return result;
}

FillInPrediction GraphSparsity::predictLUFillIn() const {
    FillInPrediction result;
    GlobalIndex n = pattern_.numRows();

    result.original_nnz = pattern_.getNnz();

    if (n == 0) {
        return result;
    }

    auto [L_pattern, U_pattern] = symbolicLU();
    result.total_factor_nnz = L_pattern.getNnz() + U_pattern.getNnz() - n;  // Subtract diagonal counted twice
    result.predicted_fill = result.total_factor_nnz - result.original_nnz;

    if (result.original_nnz > 0) {
        result.fill_ratio = static_cast<double>(result.total_factor_nnz) /
                           static_cast<double>(result.original_nnz);
    }

    return result;
}

SparsityPattern GraphSparsity::symbolicCholesky() const {
    GlobalIndex n = pattern_.numRows();

    if (n == 0) {
        return SparsityPattern(0, 0);
    }

    // Build modifiable adjacency list for elimination
    std::vector<std::set<GlobalIndex>> adj(static_cast<std::size_t>(n));
    for (GlobalIndex row = 0; row < n; ++row) {
        for (GlobalIndex col : pattern_.getRowSpan(row)) {
            if (col <= row) {
                adj[static_cast<std::size_t>(row)].insert(col);
            }
        }
    }

    // Symbolic elimination
    for (GlobalIndex k = 0; k < n; ++k) {
        // Get higher-indexed neighbors of k
        std::vector<GlobalIndex> higher_neighbors;
        for (GlobalIndex j : adj[static_cast<std::size_t>(k)]) {
            if (j > k) {
                higher_neighbors.push_back(j);
            }
        }

        // For each pair of higher neighbors, add fill-in
        for (std::size_t i = 0; i < higher_neighbors.size(); ++i) {
            GlobalIndex u = higher_neighbors[i];
            for (std::size_t j = i + 1; j < higher_neighbors.size(); ++j) {
                GlobalIndex v = higher_neighbors[j];
                // Add edge (u, v) and (v, u) in lower triangular form
                if (v < u) {
                    adj[static_cast<std::size_t>(u)].insert(v);
                } else {
                    adj[static_cast<std::size_t>(v)].insert(u);
                }
            }
        }
    }

    // Build result pattern (lower triangular)
    SparsityPattern result(n, n);
    for (GlobalIndex row = 0; row < n; ++row) {
        for (GlobalIndex col : adj[static_cast<std::size_t>(row)]) {
            result.addEntry(row, col);
        }
    }
    result.finalize();

    return result;
}

std::pair<SparsityPattern, SparsityPattern> GraphSparsity::symbolicLU() const {
    GlobalIndex n = pattern_.numRows();

    if (n == 0) {
        return {SparsityPattern(0, 0), SparsityPattern(0, 0)};
    }

    // Build modifiable adjacency
    std::vector<std::set<GlobalIndex>> L_adj(static_cast<std::size_t>(n));
    std::vector<std::set<GlobalIndex>> U_adj(static_cast<std::size_t>(n));

    for (GlobalIndex row = 0; row < n; ++row) {
        for (GlobalIndex col : pattern_.getRowSpan(row)) {
            if (col <= row) {
                L_adj[static_cast<std::size_t>(row)].insert(col);
            }
            if (col >= row) {
                U_adj[static_cast<std::size_t>(row)].insert(col);
            }
        }
    }

    // Symbolic LU elimination
    for (GlobalIndex k = 0; k < n - 1; ++k) {
        // For each row i > k where L[i,k] != 0
        for (GlobalIndex i = k + 1; i < n; ++i) {
            if (L_adj[static_cast<std::size_t>(i)].count(k) == 0) continue;

            // For each column j >= k where U[k,j] != 0
            for (GlobalIndex j : U_adj[static_cast<std::size_t>(k)]) {
                if (j < k) continue;

                if (j < i) {
                    // Fill-in in L[i,j]
                    L_adj[static_cast<std::size_t>(i)].insert(j);
                } else if (j > i) {
                    // Fill-in in U[i,j]
                    U_adj[static_cast<std::size_t>(i)].insert(j);
                }
                // j == i: diagonal, already present
            }
        }
    }

    // Build result patterns
    SparsityPattern L(n, n), U(n, n);

    for (GlobalIndex row = 0; row < n; ++row) {
        for (GlobalIndex col : L_adj[static_cast<std::size_t>(row)]) {
            L.addEntry(row, col);
        }
        for (GlobalIndex col : U_adj[static_cast<std::size_t>(row)]) {
            U.addEntry(row, col);
        }
    }

    L.finalize();
    U.finalize();

    return {std::move(L), std::move(U)};
}

// ============================================================================
// Query Methods
// ============================================================================

GlobalIndex GraphSparsity::numEdges() const {
    // For symmetric graph, NNZ includes both directions
    // Diagonal entries are self-loops
    GlobalIndex diag_count = 0;
    for (GlobalIndex i = 0; i < pattern_.numRows(); ++i) {
        if (pattern_.hasEntry(i, i)) ++diag_count;
    }

    return (pattern_.getNnz() - diag_count) / 2;
}

// ============================================================================
// Internal Helpers
// ============================================================================

GlobalIndex GraphSparsity::bfsDistance(GlobalIndex from, GlobalIndex to) const {
    if (from == to) return 0;

    GlobalIndex n = pattern_.numRows();
    std::vector<GlobalIndex> dist(static_cast<std::size_t>(n), -1);

    std::queue<GlobalIndex> queue;
    queue.push(from);
    dist[static_cast<std::size_t>(from)] = 0;

    while (!queue.empty()) {
        GlobalIndex v = queue.front();
        queue.pop();

        for (GlobalIndex neighbor : getNeighbors(v)) {
            if (dist[static_cast<std::size_t>(neighbor)] < 0) {
                dist[static_cast<std::size_t>(neighbor)] = dist[static_cast<std::size_t>(v)] + 1;
                if (neighbor == to) {
                    return dist[static_cast<std::size_t>(neighbor)];
                }
                queue.push(neighbor);
            }
        }
    }

    return -1;  // Not reachable
}

std::vector<GlobalIndex> GraphSparsity::bfsDistances(GlobalIndex from) const {
    GlobalIndex n = pattern_.numRows();
    std::vector<GlobalIndex> dist(static_cast<std::size_t>(n), -1);

    std::queue<GlobalIndex> queue;
    queue.push(from);
    dist[static_cast<std::size_t>(from)] = 0;

    while (!queue.empty()) {
        GlobalIndex v = queue.front();
        queue.pop();

        for (GlobalIndex neighbor : getNeighbors(v)) {
            if (dist[static_cast<std::size_t>(neighbor)] < 0) {
                dist[static_cast<std::size_t>(neighbor)] = dist[static_cast<std::size_t>(v)] + 1;
                queue.push(neighbor);
            }
        }
    }

    return dist;
}

// ============================================================================
// Free Functions
// ============================================================================

GlobalIndex computePatternBandwidth(const SparsityPattern& pattern) {
    return pattern.computeBandwidth();
}

GlobalIndex computePatternProfile(const SparsityPattern& pattern) {
    GlobalIndex profile = 0;

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.isFinalized()) {
            auto row_span = pattern.getRowSpan(row);
            if (!row_span.empty()) {
                profile += (row_span[row_span.size() - 1] - row_span[0] + 1);
            }
        } else {
            GlobalIndex first = -1, last = -1;
            for (GlobalIndex col = 0; col < pattern.numCols(); ++col) {
                if (pattern.hasEntry(row, col)) {
                    if (first < 0) first = col;
                    last = col;
                }
            }
            if (first >= 0) {
                profile += (last - first + 1);
            }
        }
    }

    return profile;
}

SparsityPattern applyRCM(const SparsityPattern& pattern) {
    auto perm = getRCMPermutation(pattern);
    return pattern.permute(perm, perm);
}

std::vector<GlobalIndex> getRCMPermutation(const SparsityPattern& pattern) {
    GraphSparsity graph(pattern);
    return graph.reverseCuthillMcKee();
}

ColoringResult colorPattern(const SparsityPattern& pattern) {
    GraphSparsity graph(pattern);
    return graph.degreeBasedColoring();
}

} // namespace sparsity
} // namespace FE
} // namespace svmp
