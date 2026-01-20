/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "WorkStreamAssembler.h"
#include "Dofs/DofMap.h"
#include "Dofs/DofHandler.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/ConstraintDistributor.h"
#include "Sparsity/SparsityPattern.h"
#include "Spaces/FunctionSpace.h"
#include "Elements/Element.h"

#include <chrono>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// ScratchData Implementation
// ============================================================================

ScratchData::ScratchData() = default;

ScratchData::ScratchData(LocalIndex max_dofs_per_cell, LocalIndex max_qpts, int dim)
{
    reserve(max_dofs_per_cell, max_qpts, dim);
}

ScratchData::~ScratchData() = default;

ScratchData::ScratchData(const ScratchData& other)
    : context_()  // Create new context - AssemblyContext is non-copyable
    , scratch_matrix_(other.scratch_matrix_)
    , scratch_vector_(other.scratch_vector_)
    , user_data_(other.user_data_)
{
}

ScratchData& ScratchData::operator=(const ScratchData& other)
{
    if (this != &other) {
        // context_ remains as-is (or could be cleared)
        scratch_matrix_ = other.scratch_matrix_;
        scratch_vector_ = other.scratch_vector_;
        user_data_ = other.user_data_;
    }
    return *this;
}

ScratchData::ScratchData(ScratchData&& other) noexcept = default;
ScratchData& ScratchData::operator=(ScratchData&& other) noexcept = default;

void ScratchData::reserve(LocalIndex max_dofs, LocalIndex max_qpts, int dim)
{
    context_.reserve(max_dofs, max_qpts, dim);
    scratch_matrix_.reserve(static_cast<std::size_t>(max_dofs) * max_dofs);
    scratch_vector_.reserve(max_dofs);
}

void ScratchData::clear()
{
    context_.clear();
    scratch_matrix_.clear();
    scratch_vector_.clear();
}

// ============================================================================
// CopyData Implementation
// ============================================================================

CopyData::CopyData() = default;
CopyData::~CopyData() = default;

void CopyData::reserve(LocalIndex max_dofs)
{
    const auto size = static_cast<std::size_t>(max_dofs);
    local_matrix.reserve(size * size);
    local_vector.reserve(size);
    row_dofs.reserve(size);
    col_dofs.reserve(size);
}

void CopyData::clear()
{
    local_matrix.clear();
    local_vector.clear();
    row_dofs.clear();
    col_dofs.clear();
    has_matrix = false;
    has_vector = false;
    cell_id = -1;
    is_valid_ = false;
    face_blocks.clear();
}

// ============================================================================
// CopyDataQueue Implementation
// ============================================================================

WorkStreamAssembler::CopyDataQueue::CopyDataQueue(std::size_t max_depth)
    : max_depth_(max_depth)
{
}

void WorkStreamAssembler::CopyDataQueue::push(WorkItem item)
{
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [this] {
        return queue_.size() < max_depth_ || shutdown_.load();
    });

    if (shutdown_.load()) {
        return;
    }

    queue_.push(std::move(item));

    std::size_t current_size = queue_.size();
    std::size_t expected = highwater_.load();
    while (current_size > expected &&
           !highwater_.compare_exchange_weak(expected, current_size)) {
        // Retry
    }

    not_empty_.notify_one();
}

bool WorkStreamAssembler::CopyDataQueue::tryPop(WorkItem& item)
{
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [this] {
        return !queue_.empty() || shutdown_.load();
    });

    if (queue_.empty()) {
        return false;
    }

    item = std::move(queue_.front());
    queue_.pop();
    not_full_.notify_one();
    return true;
}

void WorkStreamAssembler::CopyDataQueue::shutdown()
{
    shutdown_.store(true);
    not_full_.notify_all();
    not_empty_.notify_all();
}

// ============================================================================
// WorkStreamAssembler Construction
// ============================================================================

WorkStreamAssembler::WorkStreamAssembler()
    : ws_options_{}
{
}

WorkStreamAssembler::WorkStreamAssembler(const WorkStreamOptions& options)
    : ws_options_(options)
{
}

WorkStreamAssembler::~WorkStreamAssembler() = default;

WorkStreamAssembler::WorkStreamAssembler(WorkStreamAssembler&& other) noexcept = default;

WorkStreamAssembler& WorkStreamAssembler::operator=(WorkStreamAssembler&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void WorkStreamAssembler::setDofMap(const dofs::DofMap& dof_map)
{
    dof_map_ = &dof_map;
}

void WorkStreamAssembler::setDofHandler(const dofs::DofHandler& dof_handler)
{
    dof_handler_ = &dof_handler;
    dof_map_ = &dof_handler.getDofMap();
}

void WorkStreamAssembler::setConstraints(const constraints::AffineConstraints* constraints)
{
    constraints_ = constraints;

    if (constraints_ && constraints_->isClosed()) {
        constraint_distributor_ = std::make_unique<constraints::ConstraintDistributor>(*constraints_);
    } else {
        constraint_distributor_.reset();
    }
}

void WorkStreamAssembler::setSparsityPattern(const sparsity::SparsityPattern* sparsity)
{
    sparsity_ = sparsity;
}

void WorkStreamAssembler::setOptions(const AssemblyOptions& options)
{
    options_ = options;
    ws_options_.num_threads = (options.threading == ThreadingStrategy::Sequential) ? 1 : 4;
}

const AssemblyOptions& WorkStreamAssembler::getOptions() const noexcept
{
    return options_;
}

bool WorkStreamAssembler::isConfigured() const noexcept
{
    return dof_map_ != nullptr;
}

void WorkStreamAssembler::setWorkStreamOptions(const WorkStreamOptions& options)
{
    ws_options_ = options;
}

const WorkStreamOptions& WorkStreamAssembler::getWorkStreamOptions() const noexcept
{
    return ws_options_;
}

void WorkStreamAssembler::setScratchDataFactory(std::function<ScratchData()> factory)
{
    scratch_factory_ = std::move(factory);
}

// ============================================================================
// Lifecycle
// ============================================================================

void WorkStreamAssembler::initialize()
{
    if (!isConfigured()) {
        throw std::runtime_error("WorkStreamAssembler::initialize: not configured");
    }

    initializeScratchPool();
    initialized_ = true;
}

void WorkStreamAssembler::finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view)
{
    if (matrix_view) {
        matrix_view->endAssemblyPhase();
        matrix_view->finalizeAssembly();
    }

    if (vector_view && vector_view != matrix_view) {
        vector_view->endAssemblyPhase();
        vector_view->finalizeAssembly();
    }
}

void WorkStreamAssembler::reset()
{
    scratch_pool_.clear();
    initialized_ = false;
    last_stats_ = WorkStreamStats{};
}

// ============================================================================
// Assembly Operations
// ============================================================================

AssemblyResult WorkStreamAssembler::assembleMatrix(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view)
{
    return assembleCellsCore(mesh, test_space, trial_space, kernel,
                             &matrix_view, nullptr, true, false);
}

AssemblyResult WorkStreamAssembler::assembleVector(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView& vector_view)
{
    return assembleCellsCore(mesh, space, space, kernel,
                             nullptr, &vector_view, false, true);
}

AssemblyResult WorkStreamAssembler::assembleBoth(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView& vector_view)
{
    return assembleCellsCore(mesh, test_space, trial_space, kernel,
                             &matrix_view, &vector_view, true, true);
}

AssemblyResult WorkStreamAssembler::assembleBoundaryFaces(
    const IMeshAccess& mesh,
    int boundary_marker,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    AssemblyResult result;
    auto start_time = std::chrono::steady_clock::now();

    if (!initialized_) {
        initialize();
    }

    if (!kernel.hasBoundaryFace()) {
        return result;
    }

    if (matrix_view) matrix_view->beginAssemblyPhase();
    if (vector_view && vector_view != matrix_view) {
        vector_view->beginAssemblyPhase();
    }

    const auto required_data = kernel.getRequiredData();

    // Collect boundary faces
    std::vector<std::pair<GlobalIndex, GlobalIndex>> faces;  // (face_id, cell_id)
    mesh.forEachBoundaryFace(boundary_marker,
        [&](GlobalIndex face_id, GlobalIndex cell_id) {
            if (options_.ghost_policy != GhostPolicy::OwnedRowsOnly && !mesh.isOwnedCell(cell_id)) {
                return;
            }
            faces.emplace_back(face_id, cell_id);
        });

    // Process faces with WorkStream pattern
    std::vector<std::unique_ptr<CopyData>> all_copy_data(faces.size());

#ifdef _OPENMP
    #pragma omp parallel num_threads(ws_options_.num_threads)
    {
        int thread_id = omp_get_thread_num();
        ScratchData& scratch = *scratch_pool_[static_cast<std::size_t>(thread_id)];
        KernelOutput kernel_output;

        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < faces.size(); ++i) {
            auto [face_id, cell_id] = faces[i];

            auto copy = std::make_unique<CopyData>();
            copy->cell_id = face_id;

            // Get cell DOFs
            auto cell_dofs = dof_map_->getCellDofs(cell_id);
            copy->row_dofs.assign(cell_dofs.begin(), cell_dofs.end());
            copy->col_dofs.assign(cell_dofs.begin(), cell_dofs.end());

            // Prepare context
            LocalIndex local_face_id = 0;
            ElementType cell_type = mesh.getCellType(cell_id);
            const auto& element = space.getElement(cell_type, cell_id);
            scratch.context().configureFace(face_id, cell_id, local_face_id,
                                            element, required_data,
                                            ContextType::BoundaryFace);
            scratch.context().setBoundaryMarker(boundary_marker);

            // Compute
            kernel_output.clear();
            kernel.computeBoundaryFace(scratch.context(), boundary_marker, kernel_output);

            // Store results
            copy->has_matrix = kernel_output.has_matrix;
            copy->has_vector = kernel_output.has_vector;
            if (copy->has_matrix) {
                copy->local_matrix = kernel_output.local_matrix;
            }
            if (copy->has_vector) {
                copy->local_vector = kernel_output.local_vector;
            }
            copy->setValid(true);

            all_copy_data[i] = std::move(copy);
        }
    }
#else
    // Sequential fallback
    ScratchData& scratch = *scratch_pool_[0];
    KernelOutput kernel_output;

    for (std::size_t i = 0; i < faces.size(); ++i) {
        auto [face_id, cell_id] = faces[i];

        auto copy = std::make_unique<CopyData>();
        copy->cell_id = face_id;

        auto cell_dofs = dof_map_->getCellDofs(cell_id);
        copy->row_dofs.assign(cell_dofs.begin(), cell_dofs.end());
        copy->col_dofs.assign(cell_dofs.begin(), cell_dofs.end());

        LocalIndex local_face_id = 0;
        ElementType cell_type = mesh.getCellType(cell_id);
        const auto& element = space.getElement(cell_type, cell_id);
        scratch.context().configureFace(face_id, cell_id, local_face_id,
                                        element, required_data,
                                        ContextType::BoundaryFace);
        scratch.context().setBoundaryMarker(boundary_marker);

        kernel_output.clear();
        kernel.computeBoundaryFace(scratch.context(), boundary_marker, kernel_output);

        copy->has_matrix = kernel_output.has_matrix;
        copy->has_vector = kernel_output.has_vector;
        if (copy->has_matrix) {
            copy->local_matrix = kernel_output.local_matrix;
        }
        if (copy->has_vector) {
            copy->local_vector = kernel_output.local_vector;
        }
        copy->setValid(true);

        all_copy_data[i] = std::move(copy);
    }
#endif

    // Sequential copier (deterministic)
    runDeterministicCopier(all_copy_data, createCopier(matrix_view, vector_view));

    result.boundary_faces_assembled = static_cast<GlobalIndex>(faces.size());

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

AssemblyResult WorkStreamAssembler::assembleInteriorFaces(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView* vector_view)
{
    AssemblyResult result;
    auto start_time = std::chrono::steady_clock::now();

    if (!initialized_) {
        initialize();
    }

    if (!kernel.hasInteriorFace()) {
        return result;
    }

    matrix_view.beginAssemblyPhase();
    if (vector_view && vector_view != &matrix_view) {
        vector_view->beginAssemblyPhase();
    }

    // Interior face assembly requires collecting face data first
    // then parallel processing with DG-specific copy data

    // For now, use sequential implementation (DG face assembly is complex)
    const auto required_data = kernel.getRequiredData();
    const bool owned_rows_only = (options_.ghost_policy == GhostPolicy::OwnedRowsOnly);
    const auto should_process = [&](GlobalIndex cell_minus, GlobalIndex cell_plus) -> bool {
        if (owned_rows_only) {
            return mesh.isOwnedCell(cell_minus) || mesh.isOwnedCell(cell_plus);
        }
        return mesh.isOwnedCell(cell_minus);
    };

    mesh.forEachInteriorFace(
        [&](GlobalIndex face_id, GlobalIndex cell_minus, GlobalIndex cell_plus) {
            if (!should_process(cell_minus, cell_plus)) {
                return;
            }
            // Get DOFs
            auto minus_dofs = dof_map_->getCellDofs(cell_minus);
            auto plus_dofs = dof_map_->getCellDofs(cell_plus);

            // Prepare contexts
            ScratchData& scratch = *scratch_pool_[0];
            AssemblyContext context_plus;
            context_plus.reserve(dof_map_->getMaxDofsPerCell(), 27, mesh.dimension());

            ElementType type_minus = mesh.getCellType(cell_minus);
            ElementType type_plus = mesh.getCellType(cell_plus);
            const auto& elem_minus = test_space.getElement(type_minus, cell_minus);
            const auto& elem_plus = test_space.getElement(type_plus, cell_plus);

            scratch.context().configureFace(face_id, cell_minus, 0, elem_minus,
                                            required_data, ContextType::InteriorFace);
            context_plus.configureFace(face_id, cell_plus, 0, elem_plus,
                                       required_data, ContextType::InteriorFace);

            // Compute
            KernelOutput out_minus, out_plus, coupling_mp, coupling_pm;
            kernel.computeInteriorFace(scratch.context(), context_plus,
                                       out_minus, out_plus, coupling_mp, coupling_pm);

            // Insert all 4 blocks
            std::vector<GlobalIndex> minus_dofs_vec(minus_dofs.begin(), minus_dofs.end());
            std::vector<GlobalIndex> plus_dofs_vec(plus_dofs.begin(), plus_dofs.end());

            if (out_minus.has_matrix) {
                matrix_view.addMatrixEntries(minus_dofs_vec, minus_dofs_vec, out_minus.local_matrix);
            }
            if (out_plus.has_matrix) {
                matrix_view.addMatrixEntries(plus_dofs_vec, plus_dofs_vec, out_plus.local_matrix);
            }
            if (coupling_mp.has_matrix) {
                matrix_view.addMatrixEntries(minus_dofs_vec, plus_dofs_vec, coupling_mp.local_matrix);
            }
            if (coupling_pm.has_matrix) {
                matrix_view.addMatrixEntries(plus_dofs_vec, minus_dofs_vec, coupling_pm.local_matrix);
            }

            result.interior_faces_assembled++;
        });

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

// ============================================================================
// Generic WorkStream Interface
// ============================================================================

WorkStreamStats WorkStreamAssembler::run(
    GlobalIndex cell_begin,
    GlobalIndex cell_end,
    WorkerFunc worker,
    CopierFunc copier)
{
    WorkStreamStats stats;
    auto start_time = std::chrono::steady_clock::now();

    if (!initialized_) {
        initialize();
    }

    const GlobalIndex num_cells = cell_end - cell_begin;
    if (num_cells <= 0) {
        return stats;
    }

    // Allocate copy data for all elements
    std::vector<std::unique_ptr<CopyData>> all_copy_data(static_cast<std::size_t>(num_cells));

    auto worker_start = std::chrono::steady_clock::now();

#ifdef _OPENMP
    #pragma omp parallel num_threads(ws_options_.num_threads)
    {
        int thread_id = omp_get_thread_num();
        ScratchData& scratch = *scratch_pool_[static_cast<std::size_t>(thread_id)];

        #pragma omp for schedule(static, static_cast<int>(ws_options_.chunk_size))
        for (GlobalIndex i = 0; i < num_cells; ++i) {
            GlobalIndex cell_id = cell_begin + i;
            auto copy = std::make_unique<CopyData>();

            worker(scratch, *copy, cell_id);
            copy->cell_id = cell_id;
            copy->setValid(true);

            all_copy_data[static_cast<std::size_t>(i)] = std::move(copy);
        }
    }
#else
    ScratchData& scratch = *scratch_pool_[0];
    for (GlobalIndex i = 0; i < num_cells; ++i) {
        GlobalIndex cell_id = cell_begin + i;
        auto copy = std::make_unique<CopyData>();

        worker(scratch, *copy, cell_id);
        copy->cell_id = cell_id;
        copy->setValid(true);

        all_copy_data[static_cast<std::size_t>(i)] = std::move(copy);
    }
#endif

    auto worker_end = std::chrono::steady_clock::now();
    stats.worker_seconds = std::chrono::duration<double>(worker_end - worker_start).count();

    // Copier phase (deterministic)
    auto copier_start = std::chrono::steady_clock::now();

    if (ws_options_.deterministic_copier) {
        runDeterministicCopier(all_copy_data, copier);
    } else {
        for (auto& copy : all_copy_data) {
            if (copy && copy->isValid()) {
                copier(*copy);
            }
        }
    }

    auto copier_end = std::chrono::steady_clock::now();
    stats.copier_seconds = std::chrono::duration<double>(copier_end - copier_start).count();

    stats.elements_processed = num_cells;
    stats.total_seconds = std::chrono::duration<double>(copier_end - start_time).count();

    last_stats_ = stats;
    return stats;
}

WorkStreamStats WorkStreamAssembler::run(
    const IMeshAccess& mesh,
    WorkerFunc worker,
    CopierFunc copier)
{
    return run(0, mesh.numCells(), std::move(worker), std::move(copier));
}

const WorkStreamStats& WorkStreamAssembler::getLastStats() const noexcept
{
    return last_stats_;
}

// ============================================================================
// Internal Methods
// ============================================================================

AssemblyResult WorkStreamAssembler::assembleCellsCore(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view,
    bool assemble_matrix,
    bool assemble_vector)
{
    AssemblyResult result;

    if (!initialized_) {
        initialize();
    }

    if (matrix_view && assemble_matrix) {
        matrix_view->beginAssemblyPhase();
    }
    if (vector_view && assemble_vector && vector_view != matrix_view) {
        vector_view->beginAssemblyPhase();
    }

    const RequiredData required_data = kernel.getRequiredData();

    auto worker = createWorker(mesh, test_space, trial_space, kernel, required_data);
    auto copier = createCopier(
        assemble_matrix ? matrix_view : nullptr,
        assemble_vector ? vector_view : nullptr);

    auto stats = run(mesh, worker, copier);

    result.elements_assembled = stats.elements_processed;
    result.elapsed_time_seconds = stats.total_seconds;

    return result;
}

WorkStreamAssembler::WorkerFunc WorkStreamAssembler::createWorker(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    RequiredData required_data)
{
    return [&](ScratchData& scratch, CopyData& copy, GlobalIndex cell_id) {
        // Get DOFs
        auto test_dofs = dof_map_->getCellDofs(cell_id);
        copy.row_dofs.assign(test_dofs.begin(), test_dofs.end());
        copy.col_dofs.assign(test_dofs.begin(), test_dofs.end());

        // Prepare context
        ElementType cell_type = mesh.getCellType(cell_id);
        const auto& test_element = test_space.getElement(cell_type, cell_id);
        const auto& trial_element = trial_space.getElement(cell_type, cell_id);

        scratch.context().configure(cell_id, test_element, trial_element, required_data);

        // Compute
        KernelOutput output;
        kernel.computeCell(scratch.context(), output);

        // Store results
        copy.has_matrix = output.has_matrix;
        copy.has_vector = output.has_vector;
        if (output.has_matrix) {
            copy.local_matrix = std::move(output.local_matrix);
        }
        if (output.has_vector) {
            copy.local_vector = std::move(output.local_vector);
        }
    };
}

WorkStreamAssembler::CopierFunc WorkStreamAssembler::createCopier(
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    return [matrix_view, vector_view, this](const CopyData& copy) {
        if (!copy.isValid()) {
            return;
        }

        // Handle constraints if enabled
        bool has_constraints = ws_options_.use_constraints && constraint_distributor_;

        if (has_constraints && constraints_) {
            // Check for constrained DOFs
            bool row_constrained = constraints_->hasConstrainedDofs(copy.row_dofs);
            bool col_constrained = constraints_->hasConstrainedDofs(copy.col_dofs);

            if (row_constrained || col_constrained) {
                // Use constraint distributor
                // (Similar to StandardAssembler::insertLocalConstrained)
                // For simplicity, fall through to direct insertion
                // Real implementation would use ConstraintDistributor
            }
        }

        // Direct insertion
        if (matrix_view && copy.has_matrix) {
            matrix_view->addMatrixEntries(copy.row_dofs, copy.col_dofs, copy.local_matrix);
        }

        if (vector_view && copy.has_vector) {
            vector_view->addVectorEntries(copy.row_dofs, copy.local_vector);
        }
    };
}

void WorkStreamAssembler::initializeScratchPool()
{
    const int num_threads = ws_options_.num_threads;
    scratch_pool_.clear();
    scratch_pool_.reserve(static_cast<std::size_t>(num_threads));

    const LocalIndex max_dofs = dof_map_->getMaxDofsPerCell();
    const LocalIndex max_qpts = 27;  // Typical for Q2 hexahedra
    const int dim = 3;  // Default to 3D

    for (int i = 0; i < num_threads; ++i) {
        if (scratch_factory_) {
            scratch_pool_.push_back(std::make_unique<ScratchData>(scratch_factory_()));
        } else {
            scratch_pool_.push_back(std::make_unique<ScratchData>(max_dofs, max_qpts, dim));
        }
    }
}

void WorkStreamAssembler::runDeterministicCopier(
    const std::vector<std::unique_ptr<CopyData>>& all_copy_data,
    CopierFunc copier)
{
    // Process in order for determinism
    for (const auto& copy : all_copy_data) {
        if (copy && copy->isValid()) {
            copier(*copy);
        }
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<Assembler> createWorkStreamAssembler()
{
    return std::make_unique<WorkStreamAssembler>();
}

std::unique_ptr<Assembler> createWorkStreamAssembler(const WorkStreamOptions& options)
{
    return std::make_unique<WorkStreamAssembler>(options);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
