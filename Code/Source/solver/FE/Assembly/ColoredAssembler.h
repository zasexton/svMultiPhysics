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

#ifndef SVMP_FE_ASSEMBLY_COLORED_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_COLORED_ASSEMBLER_H

/**
 * @file ColoredAssembler.h
 * @brief Graph-colored parallel assembly strategy
 *
 * ColoredAssembler implements race-free parallel assembly using element graph
 * coloring. Elements are colored such that no two elements of the same color
 * share DOFs, allowing parallel assembly within each color without
 * synchronization or atomic operations.
 *
 * Key features:
 * - Automatic element graph construction from DOF connectivity
 * - Multiple coloring algorithms (greedy, DSatur, Kempe chain optimization)
 * - Color-wise parallel assembly with OpenMP
 * - Deterministic results (within numerical precision)
 * - Balance optimization for better load distribution
 *
 * Threading model:
 * - Colors are processed sequentially (color barrier between colors)
 * - Elements of same color are processed in parallel (no races)
 * - Thread-local scratch buffers for computation
 * - Direct insertion into global matrices (no aggregation needed)
 *
 * Determinism:
 * - Element processing order within a color is deterministic
 * - Global insertion order is determined by element ordering
 * - Results are reproducible for same input and thread count
 *
 * Performance characteristics:
 * - Coloring overhead: O(|E| * k) where k is average color degree
 * - Assembly parallelism: limited by number of colors
 * - Memory overhead: O(|E|) for color storage
 * - Ideal when: many elements, low-order elements (few colors needed)
 *
 * @see AssemblyLoop for the underlying loop with coloring support
 * @see StandardAssembler for simple sequential assembly
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Assembler.h"
#include "AssemblyKernel.h"
#include "AssemblyContext.h"
#include "GlobalSystemView.h"
#include "AssemblyLoop.h"

#include <vector>
#include <span>
#include <functional>
#include <memory>
#include <optional>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
}

namespace spaces {
    class FunctionSpace;
}

namespace constraints {
    class AffineConstraints;
}

namespace assembly {

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
    [[nodiscard]] GlobalIndex numElements() const noexcept {
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
// Colored Assembler Options
// ============================================================================

/**
 * @brief Options for ColoredAssembler
 */
struct ColoredAssemblerOptions {
    /**
     * @brief Number of threads to use (0 = auto)
     */
    int num_threads{0};

    /**
     * @brief Coloring options
     */
    ColoringOptions coloring{};

    /**
     * @brief Apply constraints during assembly
     */
    bool apply_constraints{true};

    /**
     * @brief Recompute coloring when mesh changes
     */
    bool auto_recolor{false};

    /**
     * @brief Verbose timing output
     */
    bool verbose{false};
};

// ============================================================================
// Colored Assembler
// ============================================================================

/**
 * @brief Graph-colored parallel assembler
 *
 * ColoredAssembler provides race-free parallel assembly by ensuring
 * elements processed simultaneously do not share DOFs.
 *
 * Usage:
 * @code
 *   ColoredAssembler assembler;
 *   assembler.setMesh(mesh);
 *   assembler.setDofMap(dof_map);
 *   assembler.setSpace(space);
 *
 *   // Compute coloring (one-time or when mesh changes)
 *   assembler.computeColoring();
 *
 *   // Assembly
 *   assembler.assembleMatrix(kernel, matrix_view);
 *   assembler.assembleVector(kernel, vector_view);
 * @endcode
 */
class ColoredAssembler {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    ColoredAssembler();

    /**
     * @brief Construct with options
     */
    explicit ColoredAssembler(const ColoredAssemblerOptions& options);

    /**
     * @brief Destructor
     */
    ~ColoredAssembler();

    /**
     * @brief Move constructor
     */
    ColoredAssembler(ColoredAssembler&& other) noexcept;

    /**
     * @brief Move assignment
     */
    ColoredAssembler& operator=(ColoredAssembler&& other) noexcept;

    // Non-copyable
    ColoredAssembler(const ColoredAssembler&) = delete;
    ColoredAssembler& operator=(const ColoredAssembler&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set mesh access
     */
    void setMesh(const IMeshAccess& mesh);

    /**
     * @brief Set DOF map
     */
    void setDofMap(const dofs::DofMap& dof_map);

    /**
     * @brief Set function space
     */
    void setSpace(const spaces::FunctionSpace& space);

    /**
     * @brief Set test and trial spaces (for rectangular assembly)
     */
    void setSpaces(const spaces::FunctionSpace& test_space,
                   const spaces::FunctionSpace& trial_space);

    /**
     * @brief Set constraints
     */
    void setConstraints(const constraints::AffineConstraints& constraints);

    /**
     * @brief Set assembler options
     */
    void setOptions(const ColoredAssemblerOptions& options);

    /**
     * @brief Get current options
     */
    [[nodiscard]] const ColoredAssemblerOptions& getOptions() const noexcept {
        return options_;
    }

    // =========================================================================
    // Coloring
    // =========================================================================

    /**
     * @brief Compute element coloring
     *
     * Must be called before assembly. Can be called again if mesh changes.
     *
     * @return Coloring statistics
     */
    ColoringStats computeColoring();

    /**
     * @brief Set pre-computed coloring
     *
     * @param colors Color for each element (size = numCells)
     * @param num_colors Number of colors used
     */
    void setColoring(std::span<const int> colors, int num_colors);

    /**
     * @brief Check if coloring is available
     */
    [[nodiscard]] bool hasColoring() const noexcept { return has_coloring_; }

    /**
     * @brief Get number of colors
     */
    [[nodiscard]] int numColors() const noexcept { return num_colors_; }

    /**
     * @brief Get element colors
     */
    [[nodiscard]] std::span<const int> getColors() const noexcept {
        return element_colors_;
    }

    /**
     * @brief Get coloring statistics
     */
    [[nodiscard]] const ColoringStats& getColoringStats() const noexcept {
        return coloring_stats_;
    }

    // =========================================================================
    // Assembly Operations
    // =========================================================================

    /**
     * @brief Assemble matrix using colored parallel assembly
     *
     * @param kernel Assembly kernel
     * @param matrix_view Global matrix view
     * @return Assembly statistics
     */
    LoopStatistics assembleMatrix(
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view);

    /**
     * @brief Assemble vector using colored parallel assembly
     *
     * @param kernel Assembly kernel
     * @param vector_view Global vector view
     * @return Assembly statistics
     */
    LoopStatistics assembleVector(
        AssemblyKernel& kernel,
        GlobalSystemView& vector_view);

    /**
     * @brief Assemble both matrix and vector
     *
     * @param kernel Assembly kernel
     * @param matrix_view Global matrix view
     * @param vector_view Global vector view
     * @return Assembly statistics
     */
    LoopStatistics assembleBoth(
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView& vector_view);

    /**
     * @brief Assemble with callback interface
     *
     * @param compute_callback Element computation callback
     * @param insert_callback Global insertion callback
     * @return Assembly statistics
     */
    LoopStatistics assemble(
        CellCallback compute_callback,
        CellInsertCallback insert_callback);

    // =========================================================================
    // Query
    // =========================================================================

    /**
     * @brief Check if assembler is configured
     */
    [[nodiscard]] bool isConfigured() const noexcept;

    /**
     * @brief Get last assembly statistics
     */
    [[nodiscard]] const LoopStatistics& getLastStats() const noexcept {
        return last_stats_;
    }

private:
    // =========================================================================
    // Internal Implementation
    // =========================================================================

    /**
     * @brief Build element connectivity graph
     */
    void buildElementGraph();

    /**
     * @brief Greedy coloring algorithm
     */
    int greedyColoring();

    /**
     * @brief DSatur coloring algorithm
     */
    int dsaturColoring();

    /**
     * @brief Largest-first coloring
     */
    int largestFirstColoring();

    /**
     * @brief Balance colors by reassigning elements
     */
    void balanceColors();

    /**
     * @brief Build color-wise element lists
     */
    void buildColorLists();

    /**
     * @brief Execute colored assembly loop
     */
    void executeColoredLoop(
        CellCallback& compute,
        CellInsertCallback& insert);

    // =========================================================================
    // Data Members
    // =========================================================================

    // Configuration
    ColoredAssemblerOptions options_;
    const IMeshAccess* mesh_{nullptr};
    const dofs::DofMap* dof_map_{nullptr};
    const spaces::FunctionSpace* test_space_{nullptr};
    const spaces::FunctionSpace* trial_space_{nullptr};
    const constraints::AffineConstraints* constraints_{nullptr};

    // Element graph
    ElementGraph element_graph_;

    // Coloring
    std::vector<int> element_colors_;
    int num_colors_{0};
    bool has_coloring_{false};
    ColoringStats coloring_stats_;

    // Color-wise element lists (for efficient parallel iteration)
    std::vector<std::vector<GlobalIndex>> color_elements_;

    // Assembly infrastructure
    std::unique_ptr<AssemblyLoop> loop_;

    // Thread-local storage
    std::vector<std::unique_ptr<AssemblyContext>> thread_contexts_;
    std::vector<KernelOutput> thread_outputs_;
    std::vector<std::vector<GlobalIndex>> thread_row_dofs_;
    std::vector<std::vector<GlobalIndex>> thread_col_dofs_;

    // Statistics
    LoopStatistics last_stats_;
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

#endif // SVMP_FE_ASSEMBLY_COLORED_ASSEMBLER_H
