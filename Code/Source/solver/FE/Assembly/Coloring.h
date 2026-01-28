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

#ifndef SVMP_FE_ASSEMBLY_COLORING_H
#define SVMP_FE_ASSEMBLY_COLORING_H

/**
 * @file Coloring.h
 * @brief Element graph coloring utilities (options/stats/graph + helpers)
 *
 * This header is intentionally independent of `assembly::Assembler` to avoid
 * include cycles when coloring options are embedded in `AssemblyOptions`.
 */

#include "Core/Types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
}

namespace assembly {

class IMeshAccess;

// ============================================================================
// Coloring Algorithms
// ============================================================================

/**
 * @brief Coloring algorithm selection
 */
enum class ColoringAlgorithm : std::uint8_t {
    Greedy,         ///< Simple greedy first-fit coloring
    DSatur,         ///< Saturation degree ordering (better quality)
    LargestFirst,   ///< Process largest-degree vertices first
    SmallestLast,   ///< Smallest-last vertex ordering
    BalancedGreedy  ///< Greedy with balance consideration
};

/**
 * @brief Options for element coloring
 */
struct ColoringOptions {
    ColoringAlgorithm algorithm{ColoringAlgorithm::DSatur};

    /**
     * @brief Maximum number of colors to use
     *
     * If coloring requires more colors, assembly falls back to sequential.
     */
    int max_colors{256};

    /**
     * @brief Attempt to balance color sizes
     *
     * If true, tries to distribute elements evenly across colors.
     */
    bool balance_colors{true};

    /**
     * @brief Reorder elements by color for cache efficiency
     */
    bool reorder_elements{false};

    /**
     * @brief Verbose output during coloring
     */
    bool verbose{false};
};

/**
 * @brief Statistics from coloring
 */
struct ColoringStats {
    int num_colors{0};                    ///< Number of colors used
    GlobalIndex num_elements{0};          ///< Total elements colored
    int min_color_size{0};                ///< Smallest color bucket
    int max_color_size{0};                ///< Largest color bucket
    double avg_color_size{0.0};           ///< Average color bucket size
    double coloring_seconds{0.0};         ///< Time to compute coloring
    std::vector<int> color_sizes;         ///< Size of each color bucket
};

// ============================================================================
// Element Graph
// ============================================================================

/**
 * @brief Sparse graph representing element connectivity
 *
 * Two elements are connected if they share at least one DOF.
 */
class ElementGraph {
public:
    /**
     * @brief Default constructor
     */
    ElementGraph() = default;

    /**
     * @brief Construct with number of elements
     */
    explicit ElementGraph(GlobalIndex num_elements);

    /**
     * @brief Build graph from mesh and DOF map
     */
    void build(const IMeshAccess& mesh, const dofs::DofMap& dof_map);

    /**
     * @brief Add an edge between two elements
     */
    void addEdge(GlobalIndex elem1, GlobalIndex elem2);

    /**
     * @brief Get neighbors of an element
     */
    [[nodiscard]] std::span<const GlobalIndex> neighbors(GlobalIndex elem) const;

    /**
     * @brief Get degree (number of neighbors) of an element
     */
    [[nodiscard]] int degree(GlobalIndex elem) const;

    /**
     * @brief Get maximum degree in the graph
     */
    [[nodiscard]] int maxDegree() const noexcept { return max_degree_; }

    /**
     * @brief Get number of elements
     */
    [[nodiscard]] GlobalIndex numElements() const noexcept
    {
        if (!adjacency_offsets_.empty()) {
            return static_cast<GlobalIndex>(adjacency_offsets_.size() - 1);
        }
        return static_cast<GlobalIndex>(building_adj_.size());
    }

    /**
     * @brief Get number of edges
     */
    [[nodiscard]] GlobalIndex numEdges() const noexcept { return num_edges_; }

    /**
     * @brief Clear the graph
     */
    void clear();

private:
    // CSR storage
    std::vector<GlobalIndex> adjacency_offsets_;  // Size: num_elements + 1
    std::vector<GlobalIndex> adjacency_list_;     // All neighbors concatenated

    // Building phase storage
    std::vector<std::vector<GlobalIndex>> building_adj_;

    // Stats
    int max_degree_{0};
    GlobalIndex num_edges_{0};
};

// ============================================================================
// Coloring Utilities
// ============================================================================

/**
 * @brief Compute element coloring with specified algorithm
 *
 * @param graph Element connectivity graph
 * @param algorithm Coloring algorithm to use
 * @param colors Output: color for each element
 * @return Number of colors used
 */
int colorGraph(
    const ElementGraph& graph,
    ColoringAlgorithm algorithm,
    std::vector<int>& colors);

/**
 * @brief Verify coloring is valid (no adjacent elements have same color)
 *
 * @param graph Element connectivity graph
 * @param colors Element colors
 * @return true if coloring is valid
 */
bool verifyColoring(
    const ElementGraph& graph,
    std::span<const int> colors);

/**
 * @brief Estimate optimal number of colors for a mesh
 *
 * Based on element type and mesh statistics.
 *
 * @param mesh Mesh access
 * @param dof_map DOF map
 * @return Estimated color count
 */
int estimateColorCount(
    const IMeshAccess& mesh,
    const dofs::DofMap& dof_map);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_COLORING_H

