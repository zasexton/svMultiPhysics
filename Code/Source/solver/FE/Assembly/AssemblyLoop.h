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

#ifndef SVMP_FE_ASSEMBLY_ASSEMBLY_LOOP_H
#define SVMP_FE_ASSEMBLY_ASSEMBLY_LOOP_H

/**
 * @file AssemblyLoop.h
 * @brief Unified element and face loop orchestration
 *
 * AssemblyLoop provides the core loop infrastructure for finite element assembly:
 *
 * - cell_loop: Iterate over all mesh cells and invoke kernels
 * - boundary_loop: Iterate over boundary faces for BCs
 * - interior_face_loop: Iterate over interior faces for DG methods
 *
 * Key design principles:
 * - Separation of loop orchestration from kernel execution
 * - Support for different threading strategies (sequential, colored, work-stream)
 * - Unified interface for cells and faces
 * - Deterministic iteration order for reproducibility
 *
 * Threading models supported:
 * - Sequential: Simple element-by-element iteration
 * - OpenMP parallel: Parallel loops with thread-safe insertion
 * - Colored: Race-free parallel via graph coloring
 *
 * Module boundaries:
 * - This module OWNS: loop orchestration, iteration patterns, scheduling
 * - This module does NOT OWN: kernel computation, matrix/vector storage
 *
 * Reference patterns:
 * - deal.II WorkStream: scratch/copy separation
 * - MFEM cell_loop: unified cell iteration
 * - PETSc DMPlexLoop: hierarchical entity traversal
 *
 * @see Assembler for the high-level assembly interface
 * @see AssemblyKernel for the kernel interface
 * @see AssemblyContext for per-element data
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Assembler.h"
#include "AssemblyKernel.h"
#include "AssemblyContext.h"
#include "GlobalSystemView.h"

#include <vector>
#include <span>
#include <functional>
#include <memory>
#include <optional>
#include <atomic>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
}

namespace spaces {
    class FunctionSpace;
}

namespace assembly {

// ============================================================================
// Loop Options and Configuration
// ============================================================================

/**
 * @brief Loop execution mode
 */
enum class LoopMode : std::uint8_t {
    Sequential,      ///< Simple sequential iteration
    OpenMP,          ///< OpenMP parallel for with atomic/mutex
    Colored,         ///< Graph-colored parallel (requires coloring)
    WorkStream       ///< Scratch/copy work-stream pattern
};

/**
 * @brief Options for loop execution
 */
struct LoopOptions {
    LoopMode mode{LoopMode::Sequential};
    int num_threads{1};                   ///< Number of threads (0 = auto)
    bool deterministic{true};             ///< Require deterministic order
    bool skip_ghost_cells{false};         ///< Skip ghost cells in parallel
    bool prefetch_next{false};            ///< Prefetch data for next element
    int batch_size{1};                    ///< Elements per batch (for batched kernels)
    bool verbose{false};                  ///< Print loop progress
};

/**
 * @brief Statistics from a loop execution
 */
struct LoopStatistics {
    GlobalIndex total_iterations{0};      ///< Total iterations executed
    GlobalIndex skipped_iterations{0};    ///< Iterations skipped (e.g., ghost cells)
    double elapsed_seconds{0.0};          ///< Wall-clock time
    double kernel_seconds{0.0};           ///< Time in kernel computation
    double insert_seconds{0.0};           ///< Time in global insertion
    int num_threads_used{1};              ///< Actual threads used
    std::size_t prefetch_hints{0};        ///< Number of prefetch hints issued
};

// ============================================================================
// Cell Work Item
// ============================================================================

/**
 * @brief Represents work for a single cell
 */
struct CellWorkItem {
    GlobalIndex cell_id;                  ///< Global cell identifier
    ElementType cell_type;                ///< Cell element type
    bool is_owned;                        ///< Whether cell is locally owned

    CellWorkItem() = default;
    CellWorkItem(GlobalIndex id, ElementType type, bool owned)
        : cell_id(id), cell_type(type), is_owned(owned) {}
};

/**
 * @brief Represents work for a boundary face
 */
struct BoundaryFaceWorkItem {
    GlobalIndex face_id;                  ///< Global face identifier
    GlobalIndex cell_id;                  ///< Adjacent cell identifier
    LocalIndex local_face_id;             ///< Local face index within cell
    int boundary_marker;                  ///< Boundary marker/label
    ElementType cell_type;                ///< Adjacent cell type

    BoundaryFaceWorkItem() = default;
    BoundaryFaceWorkItem(GlobalIndex fid, GlobalIndex cid, LocalIndex lfid,
                         int marker, ElementType type)
        : face_id(fid), cell_id(cid), local_face_id(lfid),
          boundary_marker(marker), cell_type(type) {}
};

/**
 * @brief Represents work for an interior face (DG)
 */
struct InteriorFaceWorkItem {
    GlobalIndex face_id;                  ///< Global face identifier
    GlobalIndex minus_cell_id;            ///< "Minus" side cell
    GlobalIndex plus_cell_id;             ///< "Plus" side cell
    LocalIndex minus_local_face_id;       ///< Local face index in minus cell
    LocalIndex plus_local_face_id;        ///< Local face index in plus cell
    ElementType minus_cell_type;          ///< Minus cell element type
    ElementType plus_cell_type;           ///< Plus cell element type

    InteriorFaceWorkItem() = default;
    InteriorFaceWorkItem(GlobalIndex fid, GlobalIndex mcid, GlobalIndex pcid,
                         LocalIndex mlfid, LocalIndex plfid,
                         ElementType mtype, ElementType ptype)
        : face_id(fid), minus_cell_id(mcid), plus_cell_id(pcid),
          minus_local_face_id(mlfid), plus_local_face_id(plfid),
          minus_cell_type(mtype), plus_cell_type(ptype) {}
};

// ============================================================================
// Cell Loop Callback Types
// ============================================================================

/**
 * @brief Callback for cell processing
 *
 * @param cell_id Cell identifier
 * @param context Prepared assembly context
 * @param output Kernel output storage
 */
using CellCallback = std::function<void(
    const CellWorkItem& cell,
    AssemblyContext& context,
    KernelOutput& output)>;

/**
 * @brief Callback for boundary face processing
 */
using BoundaryFaceCallback = std::function<void(
    const BoundaryFaceWorkItem& face,
    AssemblyContext& context,
    KernelOutput& output)>;

/**
 * @brief Callback for interior face processing (DG)
 */
using InteriorFaceCallback = std::function<void(
    const InteriorFaceWorkItem& face,
    AssemblyContext& ctx_minus,
    AssemblyContext& ctx_plus,
    KernelOutput& output_minus,
    KernelOutput& output_plus,
    KernelOutput& coupling_minus_plus,
    KernelOutput& coupling_plus_minus)>;

/**
 * @brief Callback for inserting cell contributions into global system
 */
using CellInsertCallback = std::function<void(
    const CellWorkItem& cell,
    const KernelOutput& output,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs)>;

// ============================================================================
// Assembly Loop Class
// ============================================================================

/**
 * @brief Unified loop orchestration for element and face assembly
 *
 * AssemblyLoop provides the core iteration infrastructure for FE assembly.
 * It handles:
 * - Cell iteration over the mesh
 * - Boundary face iteration for boundary conditions
 * - Interior face iteration for DG methods
 * - Threading and scheduling
 * - Context preparation and kernel invocation
 *
 * Usage:
 * @code
 *   AssemblyLoop loop;
 *   loop.setMesh(mesh_access);
 *   loop.setDofMap(dof_map);
 *   loop.setOptions(loop_options);
 *
 *   // Execute cell loop
 *   loop.cellLoop(
 *       test_space, trial_space,
 *       [&](const CellWorkItem& cell, AssemblyContext& ctx, KernelOutput& out) {
 *           kernel.computeCell(ctx, out);
 *       },
 *       [&](const CellWorkItem& cell, const KernelOutput& out,
 *           std::span<const GlobalIndex> rows, std::span<const GlobalIndex> cols) {
 *           view.addMatrixEntries(rows, cols, out.local_matrix);
 *           view.addVectorEntries(rows, out.local_vector);
 *       }
 *   );
 * @endcode
 */
class AssemblyLoop {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    AssemblyLoop();

    /**
     * @brief Construct with options
     */
    explicit AssemblyLoop(const LoopOptions& options);

    /**
     * @brief Destructor
     */
    ~AssemblyLoop();

    /**
     * @brief Move constructor
     */
    AssemblyLoop(AssemblyLoop&& other) noexcept;

    /**
     * @brief Move assignment
     */
    AssemblyLoop& operator=(AssemblyLoop&& other) noexcept;

    // Non-copyable
    AssemblyLoop(const AssemblyLoop&) = delete;
    AssemblyLoop& operator=(const AssemblyLoop&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set mesh access interface
     */
    void setMesh(const IMeshAccess& mesh);

    /**
     * @brief Set DOF map for DOF retrieval
     */
    void setDofMap(const dofs::DofMap& dof_map);

    /**
     * @brief Set loop options
     */
    void setOptions(const LoopOptions& options);

    /**
     * @brief Get current options
     */
    [[nodiscard]] const LoopOptions& getOptions() const noexcept { return options_; }

    /**
     * @brief Set element coloring for colored parallel assembly
     *
     * Required for LoopMode::Colored.
     *
     * @param colors Color assignment for each cell (size = numCells)
     * @param num_colors Number of distinct colors
     */
    void setColoring(std::span<const int> colors, int num_colors);

    /**
     * @brief Check if coloring is available
     */
    [[nodiscard]] bool hasColoring() const noexcept { return !cell_colors_.empty(); }

    // =========================================================================
    // Cell Loops
    // =========================================================================

    /**
     * @brief Execute a cell loop
     *
     * Iterates over all cells (or owned cells if skip_ghost_cells is set),
     * invoking the compute callback for each cell and then the insert callback.
     *
     * @param test_space Test function space
     * @param trial_space Trial function space
     * @param required_data Data required by the kernel
     * @param compute_callback Called to compute element contributions
     * @param insert_callback Called to insert contributions into global system
     * @return Loop statistics
     */
    LoopStatistics cellLoop(
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        RequiredData required_data,
        CellCallback compute_callback,
        CellInsertCallback insert_callback);

    /**
     * @brief Execute a cell loop (square system)
     *
     * Convenience overload for square systems.
     */
    LoopStatistics cellLoop(
        const spaces::FunctionSpace& space,
        RequiredData required_data,
        CellCallback compute_callback,
        CellInsertCallback insert_callback) {
        return cellLoop(space, space, required_data,
                        std::move(compute_callback), std::move(insert_callback));
    }

    /**
     * @brief Execute a cell loop with combined assembler
     *
     * Higher-level interface that integrates with AssemblyKernel and
     * GlobalSystemView directly.
     *
     * @param test_space Test function space
     * @param trial_space Trial function space
     * @param kernel Assembly kernel
     * @param matrix_view Matrix insertion interface (can be null)
     * @param vector_view Vector insertion interface (can be null)
     * @return Loop statistics
     */
    LoopStatistics cellLoop(
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view);

    // =========================================================================
    // Boundary Face Loops
    // =========================================================================

    /**
     * @brief Execute a boundary face loop
     *
     * Iterates over boundary faces with the specified marker.
     *
     * @param boundary_marker Boundary label to iterate
     * @param space Function space
     * @param required_data Data required by the kernel
     * @param compute_callback Called to compute face contributions
     * @param insert_callback Called to insert contributions
     * @return Loop statistics
     */
    LoopStatistics boundaryLoop(
        int boundary_marker,
        const spaces::FunctionSpace& space,
        RequiredData required_data,
        BoundaryFaceCallback compute_callback,
        CellInsertCallback insert_callback);

    /**
     * @brief Execute a boundary face loop with multiple markers
     *
     * @param boundary_markers Set of boundary markers to iterate
     * @param space Function space
     * @param required_data Data required by the kernel
     * @param compute_callback Called to compute face contributions
     * @param insert_callback Called to insert contributions
     * @return Loop statistics
     */
    LoopStatistics boundaryLoop(
        std::span<const int> boundary_markers,
        const spaces::FunctionSpace& space,
        RequiredData required_data,
        BoundaryFaceCallback compute_callback,
        CellInsertCallback insert_callback);

    /**
     * @brief Execute a boundary face loop with kernel
     *
     * @param boundary_marker Boundary label
     * @param space Function space
     * @param kernel Assembly kernel with boundary face method
     * @param matrix_view Matrix insertion (can be null)
     * @param vector_view Vector insertion (can be null)
     * @return Loop statistics
     */
    LoopStatistics boundaryLoop(
        int boundary_marker,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view);

    // =========================================================================
    // Interior Face Loops (DG)
    // =========================================================================

    /**
     * @brief Execute an interior face loop for DG methods
     *
     * Iterates over all interior faces, invoking the callback with contexts
     * for both adjacent cells.
     *
     * @param test_space Test function space
     * @param trial_space Trial function space
     * @param required_data Data required by the kernel
     * @param compute_callback Called to compute face contributions
     * @param insert_callback_minus Insert callback for minus cell contributions
     * @param insert_callback_plus Insert callback for plus cell contributions
     * @return Loop statistics
     */
    LoopStatistics interiorFaceLoop(
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        RequiredData required_data,
        InteriorFaceCallback compute_callback,
        CellInsertCallback insert_callback_minus,
        CellInsertCallback insert_callback_plus);

    /**
     * @brief Execute an interior face loop with kernel
     *
     * @param test_space Test function space
     * @param trial_space Trial function space
     * @param kernel Assembly kernel with interior face method
     * @param matrix_view Matrix insertion
     * @param vector_view Vector insertion (can be null)
     * @return Loop statistics
     */
    LoopStatistics interiorFaceLoop(
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView* vector_view);

    // =========================================================================
    // Unified Multi-Domain Loop
    // =========================================================================

    /**
     * @brief Execute combined cell, boundary, and interior face loops
     *
     * Efficient combined loop when all three domains need processing.
     *
     * @param test_space Test function space
     * @param trial_space Trial function space
     * @param kernel Assembly kernel
     * @param boundary_markers Boundary markers for boundary loop
     * @param include_interior_faces Whether to include DG interior faces
     * @param matrix_view Matrix insertion
     * @param vector_view Vector insertion (can be null)
     * @return Combined loop statistics
     */
    LoopStatistics unifiedLoop(
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        std::span<const int> boundary_markers,
        bool include_interior_faces,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view);

    // =========================================================================
    // Query
    // =========================================================================

    /**
     * @brief Get last execution statistics
     */
    [[nodiscard]] const LoopStatistics& getLastStatistics() const noexcept {
        return last_stats_;
    }

    /**
     * @brief Check if loop is properly configured
     */
    [[nodiscard]] bool isConfigured() const noexcept;

private:
    // =========================================================================
    // Internal Implementation
    // =========================================================================

    /**
     * @brief Build cell work items
     */
    void buildCellWorkItems(std::vector<CellWorkItem>& items);

    /**
     * @brief Build boundary face work items
     */
    void buildBoundaryFaceWorkItems(int marker,
                                    std::vector<BoundaryFaceWorkItem>& items);

    /**
     * @brief Build interior face work items
     */
    void buildInteriorFaceWorkItems(std::vector<InteriorFaceWorkItem>& items);

    /**
     * @brief Prepare context for a cell
     */
    void prepareContext(
        AssemblyContext& context,
        const CellWorkItem& cell,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        RequiredData required_data);

    /**
     * @brief Prepare context for a face
     */
    void prepareContextFace(
        AssemblyContext& context,
        const BoundaryFaceWorkItem& face,
        const spaces::FunctionSpace& space,
        RequiredData required_data);

    /**
     * @brief Get DOFs for a cell
     */
    void getCellDofs(
        GlobalIndex cell_id,
        const spaces::FunctionSpace& space,
        std::vector<GlobalIndex>& dofs);

    /**
     * @brief Sequential cell loop implementation
     */
    void cellLoopSequential(
        const std::vector<CellWorkItem>& items,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        RequiredData required_data,
        CellCallback& compute,
        CellInsertCallback& insert);

    /**
     * @brief OpenMP parallel cell loop implementation
     */
    void cellLoopOpenMP(
        const std::vector<CellWorkItem>& items,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        RequiredData required_data,
        CellCallback& compute,
        CellInsertCallback& insert);

    /**
     * @brief Colored parallel cell loop implementation
     */
    void cellLoopColored(
        const std::vector<CellWorkItem>& items,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        RequiredData required_data,
        CellCallback& compute,
        CellInsertCallback& insert);

    // =========================================================================
    // Data Members
    // =========================================================================

    // Configuration
    LoopOptions options_;
    const IMeshAccess* mesh_{nullptr};
    const dofs::DofMap* dof_map_{nullptr};

    // Coloring (for colored parallel)
    std::vector<int> cell_colors_;
    int num_colors_{0};

    // Thread-local storage (indexed by thread ID)
    std::vector<std::unique_ptr<AssemblyContext>> thread_contexts_;
    std::vector<KernelOutput> thread_outputs_;
    std::vector<std::vector<GlobalIndex>> thread_row_dofs_;
    std::vector<std::vector<GlobalIndex>> thread_col_dofs_;

    // Statistics
    LoopStatistics last_stats_;
};

// ============================================================================
// Free Function Loop Interfaces
// ============================================================================

/**
 * @brief Execute a simple cell loop
 *
 * Convenience function for simple cell iteration without full AssemblyLoop setup.
 *
 * @param mesh Mesh access interface
 * @param callback Called for each cell with cell_id
 * @param owned_only If true, only iterate owned cells
 */
void forEachCell(
    const IMeshAccess& mesh,
    std::function<void(GlobalIndex)> callback,
    bool owned_only = true);

/**
 * @brief Execute a boundary face loop
 *
 * @param mesh Mesh access interface
 * @param marker Boundary marker
 * @param callback Called for each face with (face_id, cell_id)
 */
void forEachBoundaryFace(
    const IMeshAccess& mesh,
    int marker,
    std::function<void(GlobalIndex, GlobalIndex)> callback);

/**
 * @brief Execute an interior face loop
 *
 * @param mesh Mesh access interface
 * @param callback Called for each face with (face_id, minus_cell_id, plus_cell_id)
 */
void forEachInteriorFace(
    const IMeshAccess& mesh,
    std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback);

// ============================================================================
// Graph Coloring for Parallel Assembly
// ============================================================================

/**
 * @brief Compute element coloring for race-free parallel assembly
 *
 * Elements are colored such that no two elements of the same color share
 * DOFs. This allows parallel assembly within each color.
 *
 * Algorithm: Greedy graph coloring on the element connectivity graph.
 *
 * @param mesh Mesh access interface
 * @param dof_map DOF map for connectivity
 * @param colors Output: color for each cell (size = numCells)
 * @return Number of colors used
 */
int computeElementColoring(
    const IMeshAccess& mesh,
    const dofs::DofMap& dof_map,
    std::vector<int>& colors);

/**
 * @brief Compute optimized coloring with minimum colors
 *
 * More expensive but produces fewer colors for better parallelism.
 *
 * @param mesh Mesh access interface
 * @param dof_map DOF map for connectivity
 * @param colors Output: color for each cell
 * @param max_iterations Maximum optimization iterations
 * @return Number of colors used
 */
int computeOptimizedColoring(
    const IMeshAccess& mesh,
    const dofs::DofMap& dof_map,
    std::vector<int>& colors,
    int max_iterations = 100);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_ASSEMBLY_LOOP_H
