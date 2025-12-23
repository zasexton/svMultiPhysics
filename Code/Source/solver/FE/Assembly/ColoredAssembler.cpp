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

#include "ColoredAssembler.h"
#include "Core/FEException.h"

#include <chrono>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// ElementGraph Implementation
// ============================================================================

ElementGraph::ElementGraph(GlobalIndex num_elements) {
    building_adj_.resize(static_cast<std::size_t>(num_elements));
}

void ElementGraph::build(const IMeshAccess& mesh, const dofs::DofMap& /*dof_map*/) {
    GlobalIndex num_cells = mesh.numCells();
    building_adj_.clear();
    building_adj_.resize(static_cast<std::size_t>(num_cells));

    // Build DOF-to-element map
    std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> dof_to_elements;

    mesh.forEachCell([this, &dof_to_elements, &mesh](GlobalIndex cell_id) {
        // Get DOFs for this cell
        // In a real implementation, this would use dof_map.getCellDofs(cell_id, dofs)
        // For now, we use node-based connectivity as a proxy

        std::vector<GlobalIndex> nodes;
        mesh.getCellNodes(cell_id, nodes);

        for (GlobalIndex node : nodes) {
            dof_to_elements[node].push_back(cell_id);
        }
    });

    // Build adjacency from shared DOFs
    num_edges_ = 0;
    max_degree_ = 0;

    for (const auto& [dof, elements] : dof_to_elements) {
        for (std::size_t i = 0; i < elements.size(); ++i) {
            for (std::size_t j = i + 1; j < elements.size(); ++j) {
                addEdge(elements[i], elements[j]);
            }
        }
    }

    // Convert to CSR format
    adjacency_offsets_.resize(static_cast<std::size_t>(num_cells) + 1);
    adjacency_offsets_[0] = 0;

    GlobalIndex total_neighbors = 0;
    for (GlobalIndex i = 0; i < num_cells; ++i) {
        auto idx = static_cast<std::size_t>(i);
        // Remove duplicates and sort
        auto& adj = building_adj_[idx];
        std::sort(adj.begin(), adj.end());
        adj.erase(std::unique(adj.begin(), adj.end()), adj.end());

        total_neighbors += static_cast<GlobalIndex>(adj.size());
        adjacency_offsets_[idx + 1] = total_neighbors;

        int deg = static_cast<int>(adj.size());
        max_degree_ = std::max(max_degree_, deg);
    }

    // Copy to flat list
    adjacency_list_.resize(static_cast<std::size_t>(total_neighbors));
    for (GlobalIndex i = 0; i < num_cells; ++i) {
        auto idx = static_cast<std::size_t>(i);
        auto offset = static_cast<std::size_t>(adjacency_offsets_[idx]);
        std::copy(building_adj_[idx].begin(), building_adj_[idx].end(),
                  adjacency_list_.begin() + static_cast<std::ptrdiff_t>(offset));
    }

    num_edges_ = total_neighbors / 2;  // Each edge counted twice

    // Clear building storage
    building_adj_.clear();
}

void ElementGraph::addEdge(GlobalIndex elem1, GlobalIndex elem2) {
    if (elem1 == elem2) return;

    auto idx1 = static_cast<std::size_t>(elem1);
    auto idx2 = static_cast<std::size_t>(elem2);

    building_adj_[idx1].push_back(elem2);
    building_adj_[idx2].push_back(elem1);
}

std::span<const GlobalIndex> ElementGraph::neighbors(GlobalIndex elem) const {
    auto idx = static_cast<std::size_t>(elem);
    auto start = static_cast<std::size_t>(adjacency_offsets_[idx]);
    auto end = static_cast<std::size_t>(adjacency_offsets_[idx + 1]);
    return {adjacency_list_.data() + start, end - start};
}

int ElementGraph::degree(GlobalIndex elem) const {
    auto idx = static_cast<std::size_t>(elem);
    return static_cast<int>(adjacency_offsets_[idx + 1] - adjacency_offsets_[idx]);
}

void ElementGraph::clear() {
    adjacency_offsets_.clear();
    adjacency_list_.clear();
    building_adj_.clear();
    max_degree_ = 0;
    num_edges_ = 0;
}

// ============================================================================
// ColoredAssembler Implementation
// ============================================================================

ColoredAssembler::ColoredAssembler()
    : options_{}
    , loop_(std::make_unique<AssemblyLoop>())
{
}

ColoredAssembler::ColoredAssembler(const ColoredAssemblerOptions& options)
    : options_(options)
    , loop_(std::make_unique<AssemblyLoop>())
{
}

ColoredAssembler::~ColoredAssembler() = default;

ColoredAssembler::ColoredAssembler(ColoredAssembler&& other) noexcept = default;

ColoredAssembler& ColoredAssembler::operator=(ColoredAssembler&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void ColoredAssembler::setMesh(const IMeshAccess& mesh) {
    mesh_ = &mesh;
    loop_->setMesh(mesh);
    has_coloring_ = false;  // Invalidate coloring when mesh changes
}

void ColoredAssembler::setDofMap(const dofs::DofMap& dof_map) {
    dof_map_ = &dof_map;
    loop_->setDofMap(dof_map);
    has_coloring_ = false;  // Invalidate coloring when DOF map changes
}

void ColoredAssembler::setSpace(const spaces::FunctionSpace& space) {
    test_space_ = &space;
    trial_space_ = &space;
}

void ColoredAssembler::setSpaces(
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space)
{
    test_space_ = &test_space;
    trial_space_ = &trial_space;
}

void ColoredAssembler::setConstraints(const constraints::AffineConstraints& constraints) {
    constraints_ = &constraints;
}

void ColoredAssembler::setOptions(const ColoredAssemblerOptions& options) {
    options_ = options;
}

bool ColoredAssembler::isConfigured() const noexcept {
    return mesh_ != nullptr &&
           dof_map_ != nullptr &&
           test_space_ != nullptr &&
           trial_space_ != nullptr;
}

// ============================================================================
// Coloring
// ============================================================================

ColoringStats ColoredAssembler::computeColoring() {
    FE_THROW_IF(!mesh_ || !dof_map_, "Mesh and DOF map must be set before coloring");

    auto start_time = std::chrono::high_resolution_clock::now();

    // Build element graph
    buildElementGraph();

    // Apply coloring algorithm
    switch (options_.coloring.algorithm) {
        case ColoringAlgorithm::Greedy:
        case ColoringAlgorithm::BalancedGreedy:
            num_colors_ = greedyColoring();
            break;
        case ColoringAlgorithm::DSatur:
            num_colors_ = dsaturColoring();
            break;
        case ColoringAlgorithm::LargestFirst:
        case ColoringAlgorithm::SmallestLast:
            num_colors_ = largestFirstColoring();
            break;
        default:
            num_colors_ = greedyColoring();
    }

    // Balance colors if requested
    if (options_.coloring.balance_colors) {
        balanceColors();
    }

    // Build color lists for efficient iteration
    buildColorLists();

    // Update loop with coloring
    loop_->setColoring(element_colors_, num_colors_);

    has_coloring_ = true;

    auto end_time = std::chrono::high_resolution_clock::now();

    // Compute statistics
    coloring_stats_.num_colors = num_colors_;
    coloring_stats_.num_elements = static_cast<GlobalIndex>(element_colors_.size());
    coloring_stats_.coloring_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    coloring_stats_.color_sizes.resize(static_cast<std::size_t>(num_colors_), 0);
    for (int color : element_colors_) {
        ++coloring_stats_.color_sizes[static_cast<std::size_t>(color)];
    }

    coloring_stats_.min_color_size = *std::min_element(
        coloring_stats_.color_sizes.begin(), coloring_stats_.color_sizes.end());
    coloring_stats_.max_color_size = *std::max_element(
        coloring_stats_.color_sizes.begin(), coloring_stats_.color_sizes.end());
    coloring_stats_.avg_color_size =
        static_cast<double>(coloring_stats_.num_elements) / num_colors_;

    if (options_.coloring.verbose) {
        // Print coloring info
    }

    return coloring_stats_;
}

void ColoredAssembler::setColoring(std::span<const int> colors, int num_colors) {
    element_colors_.assign(colors.begin(), colors.end());
    num_colors_ = num_colors;
    has_coloring_ = true;

    buildColorLists();
    loop_->setColoring(element_colors_, num_colors_);
}

void ColoredAssembler::buildElementGraph() {
    element_graph_.build(*mesh_, *dof_map_);
}

int ColoredAssembler::greedyColoring() {
    GlobalIndex n = element_graph_.numElements();
    element_colors_.resize(static_cast<std::size_t>(n), -1);

    int max_color = 0;
    std::vector<bool> used_colors(static_cast<std::size_t>(options_.coloring.max_colors), false);

    for (GlobalIndex i = 0; i < n; ++i) {
        // Find colors used by neighbors
        std::fill(used_colors.begin(), used_colors.end(), false);

        for (GlobalIndex neighbor : element_graph_.neighbors(i)) {
            int neighbor_color = element_colors_[static_cast<std::size_t>(neighbor)];
            if (neighbor_color >= 0) {
                used_colors[static_cast<std::size_t>(neighbor_color)] = true;
            }
        }

        // Find first available color
        int color = 0;
        while (color < options_.coloring.max_colors && used_colors[static_cast<std::size_t>(color)]) {
            ++color;
        }

        FE_THROW_IF(color >= options_.coloring.max_colors,
                   "Coloring requires more colors than max_colors limit");

        element_colors_[static_cast<std::size_t>(i)] = color;
        max_color = std::max(max_color, color);
    }

    return max_color + 1;
}

int ColoredAssembler::dsaturColoring() {
    GlobalIndex n = element_graph_.numElements();
    element_colors_.resize(static_cast<std::size_t>(n), -1);

    if (n == 0) return 0;

    // Saturation degree (number of different colors in neighbors)
    std::vector<int> saturation(static_cast<std::size_t>(n), 0);

    // Track which colors are used by neighbors
    std::vector<std::unordered_set<int>> neighbor_colors(static_cast<std::size_t>(n));

    // Priority queue: (saturation, degree, element_id)
    // Use negative saturation and degree for max-heap behavior
    using PQEntry = std::tuple<int, int, GlobalIndex>;
    std::priority_queue<PQEntry> pq;

    // Initialize with all vertices
    for (GlobalIndex i = 0; i < n; ++i) {
        pq.emplace(0, element_graph_.degree(i), i);
    }

    int max_color = 0;
    std::vector<bool> used_colors(static_cast<std::size_t>(options_.coloring.max_colors), false);

    while (!pq.empty()) {
        auto [sat, deg, elem] = pq.top();
        pq.pop();

        auto elem_idx = static_cast<std::size_t>(elem);

        // Skip if already colored
        if (element_colors_[elem_idx] >= 0) continue;

        // Find colors used by neighbors
        std::fill(used_colors.begin(), used_colors.end(), false);
        for (GlobalIndex neighbor : element_graph_.neighbors(elem)) {
            int neighbor_color = element_colors_[static_cast<std::size_t>(neighbor)];
            if (neighbor_color >= 0) {
                used_colors[static_cast<std::size_t>(neighbor_color)] = true;
            }
        }

        // Find first available color
        int color = 0;
        while (color < options_.coloring.max_colors && used_colors[static_cast<std::size_t>(color)]) {
            ++color;
        }

        FE_THROW_IF(color >= options_.coloring.max_colors,
                   "DSatur coloring requires more colors than max_colors limit");

        element_colors_[elem_idx] = color;
        max_color = std::max(max_color, color);

        // Update saturation of neighbors
        for (GlobalIndex neighbor : element_graph_.neighbors(elem)) {
            auto neighbor_idx = static_cast<std::size_t>(neighbor);
            if (element_colors_[neighbor_idx] < 0) {
                if (neighbor_colors[neighbor_idx].insert(color).second) {
                    ++saturation[neighbor_idx];
                    // Re-insert with updated saturation
                    pq.emplace(saturation[neighbor_idx],
                              element_graph_.degree(neighbor),
                              neighbor);
                }
            }
        }
    }

    return max_color + 1;
}

int ColoredAssembler::largestFirstColoring() {
    GlobalIndex n = element_graph_.numElements();
    element_colors_.resize(static_cast<std::size_t>(n), -1);

    if (n == 0) return 0;

    // Sort elements by degree (descending)
    std::vector<GlobalIndex> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), GlobalIndex(0));
    std::sort(order.begin(), order.end(),
              [this](GlobalIndex a, GlobalIndex b) {
                  return element_graph_.degree(a) > element_graph_.degree(b);
              });

    int max_color = 0;
    std::vector<bool> used_colors(static_cast<std::size_t>(options_.coloring.max_colors), false);

    for (GlobalIndex elem : order) {
        auto elem_idx = static_cast<std::size_t>(elem);

        // Find colors used by neighbors
        std::fill(used_colors.begin(), used_colors.end(), false);
        for (GlobalIndex neighbor : element_graph_.neighbors(elem)) {
            int neighbor_color = element_colors_[static_cast<std::size_t>(neighbor)];
            if (neighbor_color >= 0) {
                used_colors[static_cast<std::size_t>(neighbor_color)] = true;
            }
        }

        // Find first available color
        int color = 0;
        while (color < options_.coloring.max_colors && used_colors[static_cast<std::size_t>(color)]) {
            ++color;
        }

        FE_THROW_IF(color >= options_.coloring.max_colors,
                   "Largest-first coloring requires more colors than max_colors limit");

        element_colors_[elem_idx] = color;
        max_color = std::max(max_color, color);
    }

    return max_color + 1;
}

void ColoredAssembler::balanceColors() {
    // Simple balancing: try to move elements to less-populated colors

    std::vector<int> color_counts(static_cast<std::size_t>(num_colors_), 0);
    for (int color : element_colors_) {
        ++color_counts[static_cast<std::size_t>(color)];
    }

    int target_size = static_cast<int>(element_colors_.size()) / num_colors_;

    // Try to move elements from overpopulated to underpopulated colors
    for (std::size_t i = 0; i < element_colors_.size(); ++i) {
        int current_color = element_colors_[i];
        if (color_counts[static_cast<std::size_t>(current_color)] <= target_size) {
            continue;  // Already balanced
        }

        // Find which colors are valid for this element
        std::vector<bool> valid_colors(static_cast<std::size_t>(num_colors_), true);
        for (GlobalIndex neighbor : element_graph_.neighbors(static_cast<GlobalIndex>(i))) {
            int neighbor_color = element_colors_[static_cast<std::size_t>(neighbor)];
            if (neighbor_color >= 0) {
                valid_colors[static_cast<std::size_t>(neighbor_color)] = false;
            }
        }

        // Find most underpopulated valid color
        int best_color = current_color;
        int min_count = color_counts[static_cast<std::size_t>(current_color)];

        for (int c = 0; c < num_colors_; ++c) {
            if (c != current_color &&
                valid_colors[static_cast<std::size_t>(c)] &&
                color_counts[static_cast<std::size_t>(c)] < min_count - 1) {
                best_color = c;
                min_count = color_counts[static_cast<std::size_t>(c)];
            }
        }

        if (best_color != current_color) {
            --color_counts[static_cast<std::size_t>(current_color)];
            ++color_counts[static_cast<std::size_t>(best_color)];
            element_colors_[i] = best_color;
        }
    }
}

void ColoredAssembler::buildColorLists() {
    color_elements_.clear();
    color_elements_.resize(static_cast<std::size_t>(num_colors_));

    for (std::size_t i = 0; i < element_colors_.size(); ++i) {
        int color = element_colors_[i];
        color_elements_[static_cast<std::size_t>(color)].push_back(static_cast<GlobalIndex>(i));
    }
}

// ============================================================================
// Assembly Operations
// ============================================================================

LoopStatistics ColoredAssembler::assembleMatrix(
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view)
{
    FE_THROW_IF(!isConfigured(), "ColoredAssembler not configured");
    FE_THROW_IF(!has_coloring_, "Must call computeColoring() before assembly");

    // Configure loop for colored mode
    LoopOptions loop_opts;
    loop_opts.mode = LoopMode::Colored;
    loop_opts.num_threads = options_.num_threads;
    loop_opts.deterministic = true;
    loop_->setOptions(loop_opts);

    return loop_->cellLoop(*test_space_, *trial_space_, kernel, &matrix_view, nullptr);
}

LoopStatistics ColoredAssembler::assembleVector(
    AssemblyKernel& kernel,
    GlobalSystemView& vector_view)
{
    FE_THROW_IF(!isConfigured(), "ColoredAssembler not configured");
    FE_THROW_IF(!has_coloring_, "Must call computeColoring() before assembly");

    LoopOptions loop_opts;
    loop_opts.mode = LoopMode::Colored;
    loop_opts.num_threads = options_.num_threads;
    loop_opts.deterministic = true;
    loop_->setOptions(loop_opts);

    return loop_->cellLoop(*test_space_, *trial_space_, kernel, nullptr, &vector_view);
}

LoopStatistics ColoredAssembler::assembleBoth(
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView& vector_view)
{
    FE_THROW_IF(!isConfigured(), "ColoredAssembler not configured");
    FE_THROW_IF(!has_coloring_, "Must call computeColoring() before assembly");

    LoopOptions loop_opts;
    loop_opts.mode = LoopMode::Colored;
    loop_opts.num_threads = options_.num_threads;
    loop_opts.deterministic = true;
    loop_->setOptions(loop_opts);

    return loop_->cellLoop(*test_space_, *trial_space_, kernel, &matrix_view, &vector_view);
}

LoopStatistics ColoredAssembler::assemble(
    CellCallback compute_callback,
    CellInsertCallback insert_callback)
{
    FE_THROW_IF(!isConfigured(), "ColoredAssembler not configured");
    FE_THROW_IF(!has_coloring_, "Must call computeColoring() before assembly");

    LoopOptions loop_opts;
    loop_opts.mode = LoopMode::Colored;
    loop_opts.num_threads = options_.num_threads;
    loop_opts.deterministic = true;
    loop_->setOptions(loop_opts);

    RequiredData required = RequiredData::BasisValues | RequiredData::BasisGradients;

    last_stats_ = loop_->cellLoop(*test_space_, *trial_space_, required,
                                  std::move(compute_callback),
                                  std::move(insert_callback));
    return last_stats_;
}

// ============================================================================
// Utility Functions
// ============================================================================

int colorGraph(
    const ElementGraph& graph,
    ColoringAlgorithm algorithm,
    std::vector<int>& colors)
{
    GlobalIndex n = graph.numElements();
    colors.resize(static_cast<std::size_t>(n), -1);

    if (n == 0) return 0;

    // Simple greedy implementation for utility function
    int max_color = 0;
    std::vector<bool> used_colors(256, false);

    // Create ordering based on algorithm
    std::vector<GlobalIndex> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), GlobalIndex(0));

    if (algorithm == ColoringAlgorithm::LargestFirst) {
        std::sort(order.begin(), order.end(),
                  [&graph](GlobalIndex a, GlobalIndex b) {
                      return graph.degree(a) > graph.degree(b);
                  });
    }

    for (GlobalIndex elem : order) {
        auto elem_idx = static_cast<std::size_t>(elem);

        std::fill(used_colors.begin(), used_colors.end(), false);
        for (GlobalIndex neighbor : graph.neighbors(elem)) {
            int neighbor_color = colors[static_cast<std::size_t>(neighbor)];
            if (neighbor_color >= 0 && neighbor_color < 256) {
                used_colors[static_cast<std::size_t>(neighbor_color)] = true;
            }
        }

        int color = 0;
        while (color < 256 && used_colors[static_cast<std::size_t>(color)]) {
            ++color;
        }

        colors[elem_idx] = color;
        max_color = std::max(max_color, color);
    }

    return max_color + 1;
}

bool verifyColoring(
    const ElementGraph& graph,
    std::span<const int> colors)
{
    GlobalIndex n = graph.numElements();

    for (GlobalIndex i = 0; i < n; ++i) {
        int my_color = colors[static_cast<std::size_t>(i)];

        for (GlobalIndex neighbor : graph.neighbors(i)) {
            if (colors[static_cast<std::size_t>(neighbor)] == my_color) {
                return false;  // Adjacent elements have same color
            }
        }
    }

    return true;
}

int estimateColorCount(
    const IMeshAccess& /*mesh*/,
    const dofs::DofMap& /*dof_map*/)
{
    // Rough estimate based on element type
    // For tetrahedral meshes: ~20-30 colors typical
    // For hexahedral meshes: ~8-15 colors typical

    // This is a placeholder - actual implementation would analyze mesh
    return 32;  // Conservative estimate
}

} // namespace assembly
} // namespace FE
} // namespace svmp
