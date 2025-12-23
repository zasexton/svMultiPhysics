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

#include "AssemblyLoop.h"
#include "Core/FEException.h"

#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <queue>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// AssemblyLoop Implementation
// ============================================================================

AssemblyLoop::AssemblyLoop()
    : options_{}
{
}

AssemblyLoop::AssemblyLoop(const LoopOptions& options)
    : options_(options)
{
}

AssemblyLoop::~AssemblyLoop() = default;

AssemblyLoop::AssemblyLoop(AssemblyLoop&& other) noexcept = default;

AssemblyLoop& AssemblyLoop::operator=(AssemblyLoop&& other) noexcept = default;

void AssemblyLoop::setMesh(const IMeshAccess& mesh) {
    mesh_ = &mesh;
}

void AssemblyLoop::setDofMap(const dofs::DofMap& dof_map) {
    dof_map_ = &dof_map;
}

void AssemblyLoop::setOptions(const LoopOptions& options) {
    options_ = options;

    // Initialize thread-local storage based on thread count
    int num_threads = options_.num_threads;
    if (num_threads <= 0) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = 1;
#endif
    }

    thread_contexts_.resize(static_cast<std::size_t>(num_threads));
    thread_outputs_.resize(static_cast<std::size_t>(num_threads));
    thread_row_dofs_.resize(static_cast<std::size_t>(num_threads));
    thread_col_dofs_.resize(static_cast<std::size_t>(num_threads));

    for (int i = 0; i < num_threads; ++i) {
        thread_contexts_[static_cast<std::size_t>(i)] =
            std::make_unique<AssemblyContext>();
    }
}

void AssemblyLoop::setColoring(std::span<const int> colors, int num_colors) {
    cell_colors_.assign(colors.begin(), colors.end());
    num_colors_ = num_colors;
}

bool AssemblyLoop::isConfigured() const noexcept {
    return mesh_ != nullptr && dof_map_ != nullptr;
}

// ============================================================================
// Cell Loop Implementation
// ============================================================================

LoopStatistics AssemblyLoop::cellLoop(
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    RequiredData required_data,
    CellCallback compute_callback,
    CellInsertCallback insert_callback)
{
    FE_THROW_IF(!isConfigured(), "AssemblyLoop not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    // Build work items
    std::vector<CellWorkItem> items;
    buildCellWorkItems(items);

    // Execute based on mode
    switch (options_.mode) {
        case LoopMode::Sequential:
            cellLoopSequential(items, test_space, trial_space, required_data,
                               compute_callback, insert_callback);
            break;

        case LoopMode::OpenMP:
            cellLoopOpenMP(items, test_space, trial_space, required_data,
                           compute_callback, insert_callback);
            break;

        case LoopMode::Colored:
            FE_THROW_IF(!hasColoring(),
                       "Colored loop requires coloring via setColoring()");
            cellLoopColored(items, test_space, trial_space, required_data,
                            compute_callback, insert_callback);
            break;

        case LoopMode::WorkStream:
            // Work-stream uses scratch/copy pattern - fallback to sequential for now
            cellLoopSequential(items, test_space, trial_space, required_data,
                               compute_callback, insert_callback);
            break;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.elapsed_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

LoopStatistics AssemblyLoop::cellLoop(
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    RequiredData required_data = kernel.getRequiredData();
    bool need_matrix = !kernel.isVectorOnly() && matrix_view != nullptr;
    bool need_vector = !kernel.isMatrixOnly() && vector_view != nullptr;

    return cellLoop(
        test_space, trial_space, required_data,
        // Compute callback
        [&kernel, need_matrix, need_vector](
            const CellWorkItem& /*cell*/,
            AssemblyContext& context,
            KernelOutput& output) {
            LocalIndex n_test = context.numTestDofs();
            LocalIndex n_trial = context.numTrialDofs();
            output.reserve(n_test, n_trial, need_matrix, need_vector);
            output.clear();
            kernel.computeCell(context, output);
        },
        // Insert callback
        [matrix_view, vector_view, need_matrix, need_vector](
            const CellWorkItem& /*cell*/,
            const KernelOutput& output,
            std::span<const GlobalIndex> row_dofs,
            std::span<const GlobalIndex> col_dofs) {
            if (need_matrix && matrix_view != nullptr) {
                matrix_view->addMatrixEntries(row_dofs, col_dofs, output.local_matrix);
            }
            if (need_vector && vector_view != nullptr) {
                vector_view->addVectorEntries(row_dofs, output.local_vector);
            }
        }
    );
}

void AssemblyLoop::cellLoopSequential(
    const std::vector<CellWorkItem>& items,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    RequiredData required_data,
    CellCallback& compute,
    CellInsertCallback& insert)
{
    // Ensure we have at least one context
    if (thread_contexts_.empty()) {
        thread_contexts_.push_back(std::make_unique<AssemblyContext>());
        thread_outputs_.resize(1);
        thread_row_dofs_.resize(1);
        thread_col_dofs_.resize(1);
    }

    AssemblyContext& context = *thread_contexts_[0];
    KernelOutput& output = thread_outputs_[0];
    std::vector<GlobalIndex>& row_dofs = thread_row_dofs_[0];
    std::vector<GlobalIndex>& col_dofs = thread_col_dofs_[0];

    last_stats_ = LoopStatistics{};
    last_stats_.num_threads_used = 1;

    for (const auto& cell : items) {
        if (options_.skip_ghost_cells && !cell.is_owned) {
            ++last_stats_.skipped_iterations;
            continue;
        }

        // Prepare context
        prepareContext(context, cell, test_space, trial_space, required_data);

        // Get DOFs
        getCellDofs(cell.cell_id, test_space, row_dofs);
        if (&test_space != &trial_space) {
            getCellDofs(cell.cell_id, trial_space, col_dofs);
        } else {
            col_dofs = row_dofs;
        }

        // Compute
        auto kernel_start = std::chrono::high_resolution_clock::now();
        compute(cell, context, output);
        auto kernel_end = std::chrono::high_resolution_clock::now();
        last_stats_.kernel_seconds +=
            std::chrono::duration<double>(kernel_end - kernel_start).count();

        // Insert
        auto insert_start = std::chrono::high_resolution_clock::now();
        insert(cell, output, row_dofs, col_dofs);
        auto insert_end = std::chrono::high_resolution_clock::now();
        last_stats_.insert_seconds +=
            std::chrono::duration<double>(insert_end - insert_start).count();

        ++last_stats_.total_iterations;
    }
}

void AssemblyLoop::cellLoopOpenMP(
    const std::vector<CellWorkItem>& items,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    RequiredData required_data,
    CellCallback& compute,
    CellInsertCallback& insert)
{
#ifdef _OPENMP
    int num_threads = options_.num_threads > 0 ? options_.num_threads
                                               : omp_get_max_threads();
    last_stats_ = LoopStatistics{};
    last_stats_.num_threads_used = num_threads;

    // Ensure thread-local storage
    while (static_cast<int>(thread_contexts_.size()) < num_threads) {
        thread_contexts_.push_back(std::make_unique<AssemblyContext>());
        thread_outputs_.resize(thread_contexts_.size());
        thread_row_dofs_.resize(thread_contexts_.size());
        thread_col_dofs_.resize(thread_contexts_.size());
    }

    std::atomic<GlobalIndex> total_iterations{0};
    std::atomic<GlobalIndex> skipped_iterations{0};

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        AssemblyContext& context = *thread_contexts_[static_cast<std::size_t>(tid)];
        KernelOutput& output = thread_outputs_[static_cast<std::size_t>(tid)];
        std::vector<GlobalIndex>& row_dofs = thread_row_dofs_[static_cast<std::size_t>(tid)];
        std::vector<GlobalIndex>& col_dofs = thread_col_dofs_[static_cast<std::size_t>(tid)];

        #pragma omp for schedule(dynamic)
        for (std::size_t i = 0; i < items.size(); ++i) {
            const auto& cell = items[i];

            if (options_.skip_ghost_cells && !cell.is_owned) {
                ++skipped_iterations;
                continue;
            }

            // Prepare context
            prepareContext(context, cell, test_space, trial_space, required_data);

            // Get DOFs
            getCellDofs(cell.cell_id, test_space, row_dofs);
            if (&test_space != &trial_space) {
                getCellDofs(cell.cell_id, trial_space, col_dofs);
            } else {
                col_dofs = row_dofs;
            }

            // Compute
            compute(cell, context, output);

            // Insert (must be protected or use atomic operations)
            #pragma omp critical
            {
                insert(cell, output, row_dofs, col_dofs);
            }

            ++total_iterations;
        }
    }

    last_stats_.total_iterations = total_iterations.load();
    last_stats_.skipped_iterations = skipped_iterations.load();
#else
    // Fall back to sequential if OpenMP not available
    cellLoopSequential(items, test_space, trial_space, required_data,
                       compute, insert);
#endif
}

void AssemblyLoop::cellLoopColored(
    const std::vector<CellWorkItem>& items,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    RequiredData required_data,
    CellCallback& compute,
    CellInsertCallback& insert)
{
#ifdef _OPENMP
    int num_threads = options_.num_threads > 0 ? options_.num_threads
                                               : omp_get_max_threads();
    last_stats_ = LoopStatistics{};
    last_stats_.num_threads_used = num_threads;

    // Ensure thread-local storage
    while (static_cast<int>(thread_contexts_.size()) < num_threads) {
        thread_contexts_.push_back(std::make_unique<AssemblyContext>());
        thread_outputs_.resize(thread_contexts_.size());
        thread_row_dofs_.resize(thread_contexts_.size());
        thread_col_dofs_.resize(thread_contexts_.size());
    }

    // Build color-sorted indices
    std::vector<std::vector<std::size_t>> color_indices(
        static_cast<std::size_t>(num_colors_));

    for (std::size_t i = 0; i < items.size(); ++i) {
        if (options_.skip_ghost_cells && !items[i].is_owned) {
            ++last_stats_.skipped_iterations;
            continue;
        }
        int color = cell_colors_[static_cast<std::size_t>(items[i].cell_id)];
        color_indices[static_cast<std::size_t>(color)].push_back(i);
    }

    // Process each color in parallel (elements of same color don't conflict)
    for (int color = 0; color < num_colors_; ++color) {
        const auto& indices = color_indices[static_cast<std::size_t>(color)];

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            AssemblyContext& context = *thread_contexts_[static_cast<std::size_t>(tid)];
            KernelOutput& output = thread_outputs_[static_cast<std::size_t>(tid)];
            std::vector<GlobalIndex>& row_dofs = thread_row_dofs_[static_cast<std::size_t>(tid)];
            std::vector<GlobalIndex>& col_dofs = thread_col_dofs_[static_cast<std::size_t>(tid)];

            #pragma omp for schedule(static)
            for (std::size_t j = 0; j < indices.size(); ++j) {
                const auto& cell = items[indices[j]];

                // Prepare context
                prepareContext(context, cell, test_space, trial_space, required_data);

                // Get DOFs
                getCellDofs(cell.cell_id, test_space, row_dofs);
                if (&test_space != &trial_space) {
                    getCellDofs(cell.cell_id, trial_space, col_dofs);
                } else {
                    col_dofs = row_dofs;
                }

                // Compute
                compute(cell, context, output);

                // Insert (no protection needed - same color elements don't share DOFs)
                insert(cell, output, row_dofs, col_dofs);

                // Note: total_iterations update must be atomic in parallel region
            }
        }

        last_stats_.total_iterations += static_cast<GlobalIndex>(indices.size());
    }
#else
    // Fall back to sequential if OpenMP not available
    cellLoopSequential(items, test_space, trial_space, required_data,
                       compute, insert);
#endif
}

// ============================================================================
// Boundary Face Loop Implementation
// ============================================================================

LoopStatistics AssemblyLoop::boundaryLoop(
    int boundary_marker,
    const spaces::FunctionSpace& space,
    RequiredData required_data,
    BoundaryFaceCallback compute_callback,
    CellInsertCallback insert_callback)
{
    FE_THROW_IF(!isConfigured(), "AssemblyLoop not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    // Build work items
    std::vector<BoundaryFaceWorkItem> items;
    buildBoundaryFaceWorkItems(boundary_marker, items);

    // Ensure we have at least one context
    if (thread_contexts_.empty()) {
        thread_contexts_.push_back(std::make_unique<AssemblyContext>());
        thread_outputs_.resize(1);
        thread_row_dofs_.resize(1);
        thread_col_dofs_.resize(1);
    }

    AssemblyContext& context = *thread_contexts_[0];
    KernelOutput& output = thread_outputs_[0];
    std::vector<GlobalIndex>& row_dofs = thread_row_dofs_[0];
    std::vector<GlobalIndex>& col_dofs = thread_col_dofs_[0];

    last_stats_ = LoopStatistics{};
    last_stats_.num_threads_used = 1;

    for (const auto& face : items) {
        // Prepare context
        prepareContextFace(context, face, space, required_data);

        // Get DOFs
        getCellDofs(face.cell_id, space, row_dofs);
        col_dofs = row_dofs;

        // Compute
        compute_callback(face, context, output);

        // Insert using cell insert callback
        CellWorkItem dummy_cell(face.cell_id, face.cell_type, true);
        insert_callback(dummy_cell, output, row_dofs, col_dofs);

        ++last_stats_.total_iterations;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.elapsed_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

LoopStatistics AssemblyLoop::boundaryLoop(
    std::span<const int> boundary_markers,
    const spaces::FunctionSpace& space,
    RequiredData required_data,
    BoundaryFaceCallback compute_callback,
    CellInsertCallback insert_callback)
{
    LoopStatistics combined_stats{};

    for (int marker : boundary_markers) {
        auto stats = boundaryLoop(marker, space, required_data,
                                  compute_callback, insert_callback);
        combined_stats.total_iterations += stats.total_iterations;
        combined_stats.elapsed_seconds += stats.elapsed_seconds;
    }

    last_stats_ = combined_stats;
    return combined_stats;
}

LoopStatistics AssemblyLoop::boundaryLoop(
    int boundary_marker,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    FE_THROW_IF(!kernel.hasBoundaryFace(),
               "Kernel does not support boundary face computation");

    RequiredData required_data = kernel.getRequiredData();
    bool need_matrix = !kernel.isVectorOnly() && matrix_view != nullptr;
    bool need_vector = !kernel.isMatrixOnly() && vector_view != nullptr;

    return boundaryLoop(
        boundary_marker, space, required_data,
        // Compute callback
        [&kernel, need_matrix, need_vector](
            const BoundaryFaceWorkItem& face,
            AssemblyContext& context,
            KernelOutput& output) {
            LocalIndex n_dofs = context.numTestDofs();
            output.reserve(n_dofs, n_dofs, need_matrix, need_vector);
            output.clear();
            kernel.computeBoundaryFace(context, face.boundary_marker, output);
        },
        // Insert callback
        [matrix_view, vector_view, need_matrix, need_vector](
            const CellWorkItem& /*cell*/,
            const KernelOutput& output,
            std::span<const GlobalIndex> row_dofs,
            std::span<const GlobalIndex> col_dofs) {
            if (need_matrix && matrix_view != nullptr) {
                matrix_view->addMatrixEntries(row_dofs, col_dofs, output.local_matrix);
            }
            if (need_vector && vector_view != nullptr) {
                vector_view->addVectorEntries(row_dofs, output.local_vector);
            }
        }
    );
}

// ============================================================================
// Interior Face Loop Implementation
// ============================================================================

LoopStatistics AssemblyLoop::interiorFaceLoop(
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    RequiredData required_data,
    InteriorFaceCallback compute_callback,
    CellInsertCallback insert_callback_minus,
    CellInsertCallback insert_callback_plus)
{
    FE_THROW_IF(!isConfigured(), "AssemblyLoop not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    // Build work items
    std::vector<InteriorFaceWorkItem> items;
    buildInteriorFaceWorkItems(items);

    // Ensure we have contexts for both sides
    if (thread_contexts_.size() < 2) {
        while (thread_contexts_.size() < 2) {
            thread_contexts_.push_back(std::make_unique<AssemblyContext>());
        }
        thread_outputs_.resize(6);  // minus, plus, coupling x 2, and 2 extra
        thread_row_dofs_.resize(2);
        thread_col_dofs_.resize(2);
    }

    AssemblyContext& ctx_minus = *thread_contexts_[0];
    AssemblyContext& ctx_plus = *thread_contexts_[1];
    KernelOutput& output_minus = thread_outputs_[0];
    KernelOutput& output_plus = thread_outputs_[1];
    KernelOutput& coupling_minus_plus = thread_outputs_[2];
    KernelOutput& coupling_plus_minus = thread_outputs_[3];
    std::vector<GlobalIndex>& row_dofs_minus = thread_row_dofs_[0];
    std::vector<GlobalIndex>& row_dofs_plus = thread_row_dofs_[1];
    std::vector<GlobalIndex>& col_dofs_minus = thread_col_dofs_[0];
    std::vector<GlobalIndex>& col_dofs_plus = thread_col_dofs_[1];

    last_stats_ = LoopStatistics{};
    last_stats_.num_threads_used = 1;

    for (const auto& face : items) {
        // Prepare contexts for both sides
        CellWorkItem minus_cell(face.minus_cell_id, face.minus_cell_type, true);
        CellWorkItem plus_cell(face.plus_cell_id, face.plus_cell_type, true);

        prepareContext(ctx_minus, minus_cell, test_space, trial_space, required_data);
        prepareContext(ctx_plus, plus_cell, test_space, trial_space, required_data);

        // Get DOFs for both cells
        getCellDofs(face.minus_cell_id, test_space, row_dofs_minus);
        getCellDofs(face.plus_cell_id, test_space, row_dofs_plus);
        if (&test_space != &trial_space) {
            getCellDofs(face.minus_cell_id, trial_space, col_dofs_minus);
            getCellDofs(face.plus_cell_id, trial_space, col_dofs_plus);
        } else {
            col_dofs_minus = row_dofs_minus;
            col_dofs_plus = row_dofs_plus;
        }

        // Compute
        compute_callback(face, ctx_minus, ctx_plus,
                         output_minus, output_plus,
                         coupling_minus_plus, coupling_plus_minus);

        // Insert minus cell contributions
        insert_callback_minus(minus_cell, output_minus, row_dofs_minus, col_dofs_minus);

        // Insert plus cell contributions
        insert_callback_plus(plus_cell, output_plus, row_dofs_plus, col_dofs_plus);

        // Insert coupling terms
        // minus equations, plus DOFs
        insert_callback_minus(minus_cell, coupling_minus_plus, row_dofs_minus, col_dofs_plus);
        // plus equations, minus DOFs
        insert_callback_plus(plus_cell, coupling_plus_minus, row_dofs_plus, col_dofs_minus);

        ++last_stats_.total_iterations;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.elapsed_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

LoopStatistics AssemblyLoop::interiorFaceLoop(
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView* vector_view)
{
    FE_THROW_IF(!kernel.hasInteriorFace(),
               "Kernel does not support interior face computation");

    RequiredData required_data = kernel.getRequiredData();
    bool need_matrix = !kernel.isVectorOnly();
    bool need_vector = !kernel.isMatrixOnly() && vector_view != nullptr;

    return interiorFaceLoop(
        test_space, trial_space, required_data,
        // Compute callback
        [&kernel, need_matrix, need_vector](
            const InteriorFaceWorkItem& /*face*/,
            AssemblyContext& ctx_minus,
            AssemblyContext& ctx_plus,
            KernelOutput& output_minus,
            KernelOutput& output_plus,
            KernelOutput& coupling_minus_plus,
            KernelOutput& coupling_plus_minus) {
            LocalIndex n_minus = ctx_minus.numTestDofs();
            LocalIndex n_plus = ctx_plus.numTestDofs();

            output_minus.reserve(n_minus, n_minus, need_matrix, need_vector);
            output_plus.reserve(n_plus, n_plus, need_matrix, need_vector);
            coupling_minus_plus.reserve(n_minus, n_plus, need_matrix, false);
            coupling_plus_minus.reserve(n_plus, n_minus, need_matrix, false);

            output_minus.clear();
            output_plus.clear();
            coupling_minus_plus.clear();
            coupling_plus_minus.clear();

            kernel.computeInteriorFace(ctx_minus, ctx_plus,
                                       output_minus, output_plus,
                                       coupling_minus_plus, coupling_plus_minus);
        },
        // Insert callbacks
        [&matrix_view, vector_view, need_matrix, need_vector](
            const CellWorkItem& /*cell*/,
            const KernelOutput& output,
            std::span<const GlobalIndex> row_dofs,
            std::span<const GlobalIndex> col_dofs) {
            if (need_matrix) {
                matrix_view.addMatrixEntries(row_dofs, col_dofs, output.local_matrix);
            }
            if (need_vector && vector_view != nullptr) {
                vector_view->addVectorEntries(row_dofs, output.local_vector);
            }
        },
        [&matrix_view, vector_view, need_matrix, need_vector](
            const CellWorkItem& /*cell*/,
            const KernelOutput& output,
            std::span<const GlobalIndex> row_dofs,
            std::span<const GlobalIndex> col_dofs) {
            if (need_matrix) {
                matrix_view.addMatrixEntries(row_dofs, col_dofs, output.local_matrix);
            }
            if (need_vector && vector_view != nullptr) {
                vector_view->addVectorEntries(row_dofs, output.local_vector);
            }
        }
    );
}

// ============================================================================
// Unified Loop Implementation
// ============================================================================

LoopStatistics AssemblyLoop::unifiedLoop(
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    std::span<const int> boundary_markers,
    bool include_interior_faces,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    LoopStatistics combined_stats{};
    auto start_time = std::chrono::high_resolution_clock::now();

    // Cell loop
    if (kernel.hasCell()) {
        auto stats = cellLoop(test_space, trial_space, kernel, matrix_view, vector_view);
        combined_stats.total_iterations += stats.total_iterations;
        combined_stats.kernel_seconds += stats.kernel_seconds;
        combined_stats.insert_seconds += stats.insert_seconds;
    }

    // Boundary face loop
    if (kernel.hasBoundaryFace() && !boundary_markers.empty()) {
        for (int marker : boundary_markers) {
            auto stats = boundaryLoop(marker, test_space, kernel, matrix_view, vector_view);
            combined_stats.total_iterations += stats.total_iterations;
            combined_stats.kernel_seconds += stats.kernel_seconds;
            combined_stats.insert_seconds += stats.insert_seconds;
        }
    }

    // Interior face loop (DG)
    if (include_interior_faces && kernel.hasInteriorFace() && matrix_view != nullptr) {
        auto stats = interiorFaceLoop(test_space, trial_space, kernel,
                                      *matrix_view, vector_view);
        combined_stats.total_iterations += stats.total_iterations;
        combined_stats.kernel_seconds += stats.kernel_seconds;
        combined_stats.insert_seconds += stats.insert_seconds;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    combined_stats.elapsed_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    last_stats_ = combined_stats;
    return combined_stats;
}

// ============================================================================
// Internal Helpers
// ============================================================================

void AssemblyLoop::buildCellWorkItems(std::vector<CellWorkItem>& items) {
    items.clear();
    items.reserve(static_cast<std::size_t>(mesh_->numCells()));

    mesh_->forEachCell([this, &items](GlobalIndex cell_id) {
        bool owned = mesh_->isOwnedCell(cell_id);
        ElementType type = mesh_->getCellType(cell_id);
        items.emplace_back(cell_id, type, owned);
    });
}

void AssemblyLoop::buildBoundaryFaceWorkItems(
    int marker,
    std::vector<BoundaryFaceWorkItem>& items)
{
    items.clear();

    mesh_->forEachBoundaryFace(marker,
        [this, marker, &items](GlobalIndex face_id, GlobalIndex cell_id) {
            ElementType cell_type = mesh_->getCellType(cell_id);
            // Note: local_face_id would need mesh support - using 0 as placeholder
            items.emplace_back(face_id, cell_id, 0, marker, cell_type);
        }
    );
}

void AssemblyLoop::buildInteriorFaceWorkItems(std::vector<InteriorFaceWorkItem>& items) {
    items.clear();

    mesh_->forEachInteriorFace(
        [this, &items](GlobalIndex face_id, GlobalIndex minus_id, GlobalIndex plus_id) {
            ElementType minus_type = mesh_->getCellType(minus_id);
            ElementType plus_type = mesh_->getCellType(plus_id);
            // Note: local face IDs would need mesh support
            items.emplace_back(face_id, minus_id, plus_id, 0, 0, minus_type, plus_type);
        }
    );
}

void AssemblyLoop::prepareContext(
    AssemblyContext& /*context*/,
    const CellWorkItem& /*cell*/,
    const spaces::FunctionSpace& /*test_space*/,
    const spaces::FunctionSpace& /*trial_space*/,
    RequiredData /*required_data*/)
{
    // Context preparation would involve:
    // 1. Getting element from space
    // 2. Evaluating basis functions at quadrature points
    // 3. Computing geometry (Jacobians, physical points)
    // 4. Storing data in context

    // This is a placeholder - full implementation requires FunctionSpace and Element APIs
    // The actual implementation would call:
    // - space.getElement(cell_type)
    // - element.evaluateBasis(...)
    // - mapping.computeJacobian(...)
    // - context.setTestBasisData(...)
    // etc.
}

void AssemblyLoop::prepareContextFace(
    AssemblyContext& /*context*/,
    const BoundaryFaceWorkItem& /*face*/,
    const spaces::FunctionSpace& /*space*/,
    RequiredData /*required_data*/)
{
    // Similar to prepareContext but for face quadrature
    // Would compute face normals, surface quadrature, etc.
}

void AssemblyLoop::getCellDofs(
    GlobalIndex /*cell_id*/,
    const spaces::FunctionSpace& /*space*/,
    std::vector<GlobalIndex>& dofs)
{
    // This would use DofMap to get DOFs for the cell
    // dof_map_->getCellDofs(cell_id, space, dofs);

    // Placeholder - return empty for now
    dofs.clear();
}

// ============================================================================
// Free Function Implementations
// ============================================================================

void forEachCell(
    const IMeshAccess& mesh,
    std::function<void(GlobalIndex)> callback,
    bool owned_only)
{
    if (owned_only) {
        mesh.forEachOwnedCell(std::move(callback));
    } else {
        mesh.forEachCell(std::move(callback));
    }
}

void forEachBoundaryFace(
    const IMeshAccess& mesh,
    int marker,
    std::function<void(GlobalIndex, GlobalIndex)> callback)
{
    mesh.forEachBoundaryFace(marker, std::move(callback));
}

void forEachInteriorFace(
    const IMeshAccess& mesh,
    std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback)
{
    mesh.forEachInteriorFace(std::move(callback));
}

// ============================================================================
// Graph Coloring Implementation
// ============================================================================

int computeElementColoring(
    const IMeshAccess& mesh,
    const dofs::DofMap& /*dof_map*/,
    std::vector<int>& colors)
{
    GlobalIndex num_cells = mesh.numCells();
    colors.resize(static_cast<std::size_t>(num_cells), -1);

    if (num_cells == 0) {
        return 0;
    }

    // Build element connectivity graph
    // Two elements are connected if they share DOFs

    // For now, use a simple greedy coloring
    // A full implementation would build the actual connectivity

    std::vector<std::unordered_set<int>> neighbor_colors(static_cast<std::size_t>(num_cells));
    int max_color = 0;

    mesh.forEachCell([&colors, &neighbor_colors, &max_color](GlobalIndex cell_id) {
        auto idx = static_cast<std::size_t>(cell_id);

        // Find smallest available color
        int color = 0;
        while (neighbor_colors[idx].count(color) > 0) {
            ++color;
        }

        colors[idx] = color;
        max_color = std::max(max_color, color);

        // In a full implementation, we would update neighbor_colors
        // for all neighboring elements. This requires connectivity info.
    });

    return max_color + 1;
}

int computeOptimizedColoring(
    const IMeshAccess& mesh,
    const dofs::DofMap& dof_map,
    std::vector<int>& colors,
    int /*max_iterations*/)
{
    // Start with greedy coloring
    int num_colors = computeElementColoring(mesh, dof_map, colors);

    // Optimization could involve:
    // - Kempe chain improvements
    // - DSatur algorithm
    // - Simulated annealing

    // For now, return the greedy result
    return num_colors;
}

} // namespace assembly
} // namespace FE
} // namespace svmp
