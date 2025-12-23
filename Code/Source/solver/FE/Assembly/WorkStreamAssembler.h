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

#ifndef SVMP_FE_ASSEMBLY_WORKSTREAM_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_WORKSTREAM_ASSEMBLER_H

/**
 * @file WorkStreamAssembler.h
 * @brief Task-based thread-parallel assembly with Scratch/Copy pattern
 *
 * WorkStreamAssembler implements the "WorkStream" pattern (following deal.II's
 * approach) for deterministic parallel finite element assembly. The key insight
 * is separating the work into two phases:
 *
 * 1. WORKER PHASE (parallel, thread-safe):
 *    - Each worker thread has private ScratchData
 *    - Workers compute local element matrices/vectors
 *    - Results stored in CopyData (per-element output container)
 *    - No global state modification - pure computation
 *
 * 2. COPIER PHASE (sequential, deterministic):
 *    - Single thread processes CopyData objects in element order
 *    - Performs actual insertion into global system
 *    - Deterministic reduction ensures bit-reproducible results
 *
 * Benefits:
 * - No race conditions during insertion
 * - Deterministic floating-point results
 * - Good cache behavior due to per-thread scratch
 * - Works with any backend that supports sequential insertion
 *
 * Memory model:
 * - ScratchData: reusable per-thread workspace (FE values, Jacobians, etc.)
 * - CopyData: per-element results queue, consumed by copier
 *
 * Threading model:
 * - Workers run in parallel (OpenMP, TBB, or std::thread)
 * - Copier runs sequentially OR with per-row locking for finer grain
 * - Element ordering preserved for determinism
 *
 * @see ColoredAssembler for alternative race-free approach (less determinism)
 * @see StandardAssembler for sequential reference implementation
 */

#include "Core/Types.h"
#include "Assembler.h"
#include "AssemblyKernel.h"
#include "AssemblyContext.h"
#include "GlobalSystemView.h"

#include <vector>
#include <queue>
#include <memory>
#include <functional>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
    class DofHandler;
}

namespace spaces {
    class FunctionSpace;
}

namespace sparsity {
    class SparsityPattern;
}

namespace constraints {
    class AffineConstraints;
    class ConstraintDistributor;
}

namespace assembly {

// ============================================================================
// Scratch and Copy Data Structures
// ============================================================================

/**
 * @brief Per-thread scratch data for WorkStream assembly
 *
 * ScratchData contains all temporary data needed to compute element
 * contributions. Each worker thread owns one instance, avoiding allocation
 * during the assembly loop.
 *
 * Data includes:
 * - AssemblyContext (FE values, gradients, Jacobians)
 * - Solution values at quadrature points
 * - Material properties
 * - Temporary matrices/vectors for computation
 *
 * Users can extend this class for problem-specific scratch data.
 */
class ScratchData {
public:
    ScratchData();
    explicit ScratchData(LocalIndex max_dofs_per_cell, LocalIndex max_qpts, int dim);
    ~ScratchData();

    ScratchData(const ScratchData& other);
    ScratchData& operator=(const ScratchData& other);

    ScratchData(ScratchData&& other) noexcept;
    ScratchData& operator=(ScratchData&& other) noexcept;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Reserve storage based on expected sizes
     */
    void reserve(LocalIndex max_dofs, LocalIndex max_qpts, int dim);

    /**
     * @brief Clear all data (but keep allocated capacity)
     */
    void clear();

    // =========================================================================
    // Data Access
    // =========================================================================

    /**
     * @brief Get assembly context
     */
    [[nodiscard]] AssemblyContext& context() noexcept { return context_; }
    [[nodiscard]] const AssemblyContext& context() const noexcept { return context_; }

    /**
     * @brief Get/set scratch matrices
     */
    [[nodiscard]] std::vector<Real>& scratchMatrix() noexcept { return scratch_matrix_; }
    [[nodiscard]] const std::vector<Real>& scratchMatrix() const noexcept { return scratch_matrix_; }

    /**
     * @brief Get/set scratch vectors
     */
    [[nodiscard]] std::vector<Real>& scratchVector() noexcept { return scratch_vector_; }
    [[nodiscard]] const std::vector<Real>& scratchVector() const noexcept { return scratch_vector_; }

    /**
     * @brief User-defined extension data
     */
    template<typename T>
    void setUserData(std::shared_ptr<T> data) {
        user_data_ = std::static_pointer_cast<void>(data);
    }

    template<typename T>
    [[nodiscard]] T* getUserData() {
        return static_cast<T*>(user_data_.get());
    }

private:
    AssemblyContext context_;
    std::vector<Real> scratch_matrix_;
    std::vector<Real> scratch_vector_;
    std::shared_ptr<void> user_data_;
};

/**
 * @brief Per-element copy data for WorkStream assembly
 *
 * CopyData holds the results from processing one or more elements.
 * After a worker computes an element's contributions, it stores them
 * in CopyData which is then passed to the copier.
 *
 * Each CopyData instance contains:
 * - Local element matrix
 * - Local element vector
 * - DOF indices for row and column mapping
 * - Element identifier for debugging/logging
 *
 * For face assembly (DG), CopyData can hold multiple blocks
 * (self-coupling and cross-coupling terms).
 */
class CopyData {
public:
    CopyData();
    ~CopyData();

    CopyData(const CopyData& other) = default;
    CopyData& operator=(const CopyData& other) = default;

    CopyData(CopyData&& other) noexcept = default;
    CopyData& operator=(CopyData&& other) noexcept = default;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Reserve storage
     */
    void reserve(LocalIndex max_dofs);

    /**
     * @brief Clear for reuse
     */
    void clear();

    /**
     * @brief Check if data is valid/populated
     */
    [[nodiscard]] bool isValid() const noexcept { return is_valid_; }
    void setValid(bool valid) noexcept { is_valid_ = valid; }

    // =========================================================================
    // Cell Data
    // =========================================================================

    /**
     * @brief Local element matrix (row-major)
     */
    std::vector<Real> local_matrix;

    /**
     * @brief Local element vector
     */
    std::vector<Real> local_vector;

    /**
     * @brief Row DOF indices (for insertion)
     */
    std::vector<GlobalIndex> row_dofs;

    /**
     * @brief Column DOF indices (for insertion)
     */
    std::vector<GlobalIndex> col_dofs;

    /**
     * @brief Flags indicating what's computed
     */
    bool has_matrix{false};
    bool has_vector{false};

    /**
     * @brief Element/face identifier
     */
    GlobalIndex cell_id{-1};

    // =========================================================================
    // Face Data (for DG)
    // =========================================================================

    /**
     * @brief Additional blocks for interior face terms
     */
    struct FaceBlock {
        std::vector<Real> matrix;
        std::vector<GlobalIndex> row_dofs;
        std::vector<GlobalIndex> col_dofs;
    };

    std::vector<FaceBlock> face_blocks;

private:
    bool is_valid_{false};
};

// ============================================================================
// WorkStream Options
// ============================================================================

/**
 * @brief Configuration options for WorkStream assembly
 */
struct WorkStreamOptions {
    /**
     * @brief Number of worker threads
     */
    int num_threads{4};

    /**
     * @brief Size of work queue (elements per chunk)
     */
    std::size_t chunk_size{64};

    /**
     * @brief Maximum pending CopyData items in queue
     *
     * Limits memory usage when copier is slower than workers.
     */
    std::size_t max_queue_depth{1024};

    /**
     * @brief Use deterministic copier (sequential in element order)
     *
     * When true, copier processes elements in strict order for reproducibility.
     * When false, copier may process out-of-order for better throughput.
     */
    bool deterministic_copier{true};

    /**
     * @brief Enable profiling/timing
     */
    bool enable_profiling{false};

    /**
     * @brief Use constraints during assembly
     */
    bool use_constraints{true};
};

/**
 * @brief Statistics from WorkStream assembly
 */
struct WorkStreamStats {
    GlobalIndex elements_processed{0};
    GlobalIndex faces_processed{0};
    double total_seconds{0.0};
    double worker_seconds{0.0};  // Total worker time (summed across threads)
    double copier_seconds{0.0};
    std::size_t queue_highwater{0};  // Max queue depth reached
};

// ============================================================================
// WorkStreamAssembler
// ============================================================================

/**
 * @brief Task-based parallel assembler with deterministic reduction
 *
 * WorkStreamAssembler uses the Scratch/Copy pattern for thread-parallel
 * assembly with deterministic results.
 *
 * Usage:
 * @code
 *   WorkStreamAssembler assembler(options);
 *   assembler.setDofMap(dof_map);
 *   assembler.setConstraints(&constraints);
 *   assembler.setSparsityPattern(&sparsity);
 *
 *   // Matrix assembly
 *   auto result = assembler.assembleMatrix(mesh, test_space, trial_space,
 *                                          kernel, matrix_view);
 *
 *   // Or use run() with explicit worker/copier functions
 *   assembler.run(mesh, space, kernel,
 *       [](ScratchData& scratch, CopyData& copy, GlobalIndex cell_id) {
 *           // Worker: compute local contributions
 *       },
 *       [](const CopyData& copy) {
 *           // Copier: insert into global system
 *       });
 * @endcode
 */
class WorkStreamAssembler : public Assembler {
public:
    // =========================================================================
    // Types
    // =========================================================================

    using WorkerFunc = std::function<void(ScratchData&, CopyData&, GlobalIndex cell_id)>;
    using CopierFunc = std::function<void(const CopyData&)>;

    // =========================================================================
    // Construction
    // =========================================================================

    WorkStreamAssembler();
    explicit WorkStreamAssembler(const WorkStreamOptions& options);
    ~WorkStreamAssembler() override;

    WorkStreamAssembler(WorkStreamAssembler&& other) noexcept;
    WorkStreamAssembler& operator=(WorkStreamAssembler&& other) noexcept;

    // Non-copyable
    WorkStreamAssembler(const WorkStreamAssembler&) = delete;
    WorkStreamAssembler& operator=(const WorkStreamAssembler&) = delete;

    // =========================================================================
    // Configuration (Assembler interface)
    // =========================================================================

    void setDofMap(const dofs::DofMap& dof_map) override;
    void setDofHandler(const dofs::DofHandler& dof_handler) override;
    void setConstraints(const constraints::AffineConstraints* constraints) override;
    void setSparsityPattern(const sparsity::SparsityPattern* sparsity) override;
    void setOptions(const AssemblyOptions& options) override;

    [[nodiscard]] const AssemblyOptions& getOptions() const noexcept override;
    [[nodiscard]] bool isConfigured() const noexcept override;
    [[nodiscard]] std::string name() const override { return "WorkStreamAssembler"; }

    // =========================================================================
    // WorkStream-Specific Configuration
    // =========================================================================

    /**
     * @brief Set WorkStream-specific options
     */
    void setWorkStreamOptions(const WorkStreamOptions& options);

    /**
     * @brief Get current WorkStream options
     */
    [[nodiscard]] const WorkStreamOptions& getWorkStreamOptions() const noexcept;

    /**
     * @brief Register custom scratch data factory
     *
     * Called to create ScratchData for each worker thread.
     */
    void setScratchDataFactory(std::function<ScratchData()> factory);

    // =========================================================================
    // Assembly Operations (Assembler interface)
    // =========================================================================

    void initialize() override;
    void finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view) override;
    void reset() override;

    AssemblyResult assembleMatrix(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view) override;

    AssemblyResult assembleVector(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView& vector_view) override;

    AssemblyResult assembleBoth(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView& vector_view) override;

    AssemblyResult assembleBoundaryFaces(
        const IMeshAccess& mesh,
        int boundary_marker,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view) override;

    AssemblyResult assembleInteriorFaces(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView* vector_view) override;

    // =========================================================================
    // Generic WorkStream Interface
    // =========================================================================

    /**
     * @brief Run WorkStream with custom worker and copier
     *
     * Most flexible interface - user provides worker and copier functions.
     *
     * @param cell_range Range of cell IDs to process [begin, end)
     * @param worker Function to compute element contributions
     * @param copier Function to insert into global system
     * @return Assembly statistics
     */
    WorkStreamStats run(
        GlobalIndex cell_begin,
        GlobalIndex cell_end,
        WorkerFunc worker,
        CopierFunc copier);

    /**
     * @brief Run over all cells in mesh
     */
    WorkStreamStats run(
        const IMeshAccess& mesh,
        WorkerFunc worker,
        CopierFunc copier);

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Get statistics from last assembly
     */
    [[nodiscard]] const WorkStreamStats& getLastStats() const noexcept;

private:
    // =========================================================================
    // Internal Types
    // =========================================================================

    /**
     * @brief Work item in the pipeline
     */
    struct WorkItem {
        GlobalIndex cell_id;
        std::unique_ptr<CopyData> copy_data;
    };

    /**
     * @brief Thread-safe queue for passing CopyData to copier
     */
    class CopyDataQueue {
    public:
        explicit CopyDataQueue(std::size_t max_depth);

        void push(WorkItem item);
        bool tryPop(WorkItem& item);
        void shutdown();
        [[nodiscard]] std::size_t highwater() const { return highwater_.load(); }

    private:
        std::queue<WorkItem> queue_;
        std::mutex mutex_;
        std::condition_variable not_full_;
        std::condition_variable not_empty_;
        std::size_t max_depth_;
        std::atomic<std::size_t> highwater_{0};
        std::atomic<bool> shutdown_{false};
    };

    // =========================================================================
    // Internal Methods
    // =========================================================================

    /**
     * @brief Core assembly implementation
     */
    AssemblyResult assembleCellsCore(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view,
        bool assemble_matrix,
        bool assemble_vector);

    /**
     * @brief Create worker function for standard assembly
     */
    WorkerFunc createWorker(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        RequiredData required_data);

    /**
     * @brief Create copier function for standard assembly
     */
    CopierFunc createCopier(
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view);

    /**
     * @brief Initialize thread-local scratch data
     */
    void initializeScratchPool();

    /**
     * @brief Run the deterministic copier (in-order)
     */
    void runDeterministicCopier(
        const std::vector<std::unique_ptr<CopyData>>& all_copy_data,
        CopierFunc copier);

    // =========================================================================
    // Data Members
    // =========================================================================

    WorkStreamOptions ws_options_;
    AssemblyOptions options_;

    const dofs::DofMap* dof_map_{nullptr};
    const dofs::DofHandler* dof_handler_{nullptr};
    const constraints::AffineConstraints* constraints_{nullptr};
    const sparsity::SparsityPattern* sparsity_{nullptr};
    std::unique_ptr<constraints::ConstraintDistributor> constraint_distributor_;

    // Thread pool scratch data
    std::vector<std::unique_ptr<ScratchData>> scratch_pool_;
    std::function<ScratchData()> scratch_factory_;

    // Statistics
    WorkStreamStats last_stats_;

    bool initialized_{false};
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create WorkStream assembler with default options
 */
std::unique_ptr<Assembler> createWorkStreamAssembler();

/**
 * @brief Create WorkStream assembler with specified options
 */
std::unique_ptr<Assembler> createWorkStreamAssembler(const WorkStreamOptions& options);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_WORKSTREAM_ASSEMBLER_H
