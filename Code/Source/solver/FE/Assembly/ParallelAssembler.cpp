/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "ParallelAssembler.h"
#include "Dofs/DofMap.h"
#include "Dofs/DofHandler.h"
#include "Dofs/GhostDofManager.h"
#include "Constraints/AffineConstraints.h"
#include "Sparsity/SparsityPattern.h"
#include "Spaces/FunctionSpace.h"
#include "Elements/Element.h"

#include <chrono>
#include <algorithm>

namespace svmp {
namespace FE {
namespace assembly {

#if FE_HAS_MPI
namespace {
void set_mpi_rank_and_size(MPI_Comm comm, int& rank, int& size)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    } else {
        rank = 0;
        size = 1;
    }
}
} // namespace
#endif

// ============================================================================
// Construction
// ============================================================================

ParallelAssembler::ParallelAssembler()
{
#if FE_HAS_MPI
    set_mpi_rank_and_size(comm_, my_rank_, world_size_);
#endif
}

#if FE_HAS_MPI
ParallelAssembler::ParallelAssembler(MPI_Comm comm)
    : comm_(comm)
{
    set_mpi_rank_and_size(comm_, my_rank_, world_size_);
}
#endif

ParallelAssembler::ParallelAssembler(const AssemblyOptions& options)
    : options_(options)
    , ghost_policy_(options.ghost_policy)
    , overlap_comm_(options.overlap_communication)
{
#if FE_HAS_MPI
    set_mpi_rank_and_size(comm_, my_rank_, world_size_);
#endif
}

ParallelAssembler::~ParallelAssembler() = default;

ParallelAssembler::ParallelAssembler(ParallelAssembler&& other) noexcept = default;

ParallelAssembler& ParallelAssembler::operator=(ParallelAssembler&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void ParallelAssembler::setDofMap(const dofs::DofMap& dof_map)
{
    dof_map_ = &dof_map;
    local_assembler_.setDofMap(dof_map);
    ghost_manager_.setDofMap(dof_map);
}

void ParallelAssembler::setDofHandler(const dofs::DofHandler& dof_handler)
{
    dof_handler_ = &dof_handler;
    dof_map_ = &dof_handler.getDofMap();
    local_assembler_.setDofHandler(dof_handler);
    ghost_manager_.setDofMap(*dof_map_);

    // Also set ghost DOF manager if available
    if (dof_handler.getGhostManager()) {
        setGhostDofManager(*dof_handler.getGhostManager());
    }
}

void ParallelAssembler::setConstraints(const constraints::AffineConstraints* constraints)
{
    constraints_ = constraints;
    local_assembler_.setConstraints(constraints);
}

void ParallelAssembler::setSparsityPattern(const sparsity::SparsityPattern* sparsity)
{
    sparsity_ = sparsity;
    local_assembler_.setSparsityPattern(sparsity);
}

void ParallelAssembler::setOptions(const AssemblyOptions& options)
{
    options_ = options;
    ghost_policy_ = options.ghost_policy;
    overlap_comm_ = options.overlap_communication;
    local_assembler_.setOptions(options);
}

const AssemblyOptions& ParallelAssembler::getOptions() const noexcept
{
    return options_;
}

#if FE_HAS_MPI
void ParallelAssembler::setComm(MPI_Comm comm)
{
    comm_ = comm;
    set_mpi_rank_and_size(comm_, my_rank_, world_size_);
    ghost_manager_.setComm(comm);
}
#endif

void ParallelAssembler::setGhostDofManager(const dofs::GhostDofManager& ghost_manager)
{
    ghost_dof_manager_ = &ghost_manager;
    ghost_manager_.setGhostDofManager(ghost_manager);
}

void ParallelAssembler::setGhostPolicy(GhostPolicy policy)
{
    ghost_policy_ = policy;
    ghost_manager_.setPolicy(policy);
}

bool ParallelAssembler::isConfigured() const noexcept
{
    return dof_map_ != nullptr;
}

// ============================================================================
// Lifecycle
// ============================================================================

void ParallelAssembler::initialize()
{
    if (!isConfigured()) {
        throw std::runtime_error("ParallelAssembler::initialize: not configured");
    }

    // Initialize local assembler
    local_assembler_.initialize();

    // Initialize ghost manager
    ghost_manager_.setPolicy(ghost_policy_);
    ghost_manager_.setDeterministic(options_.deterministic);
    ghost_manager_.initialize();

    // Reserve working storage
    const auto max_dofs = dof_map_->getMaxDofsPerCell();
    row_dofs_.reserve(static_cast<std::size_t>(max_dofs));
    col_dofs_.reserve(static_cast<std::size_t>(max_dofs));
    owned_contributions_.reserve(static_cast<std::size_t>(max_dofs * max_dofs));

    // Estimate ghost contributions
    ghost_manager_.reserveBuffers(1000);  // Heuristic

    initialized_ = true;
}

void ParallelAssembler::finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view)
{
    // Exchange ghost contributions
    if (ghost_policy_ == GhostPolicy::ReverseScatter) {
        exchangeGhostContributions();
        applyReceivedContributions(matrix_view, vector_view);
    }

    // End assembly phases
    if (matrix_view) {
        matrix_view->endAssemblyPhase();
        matrix_view->finalizeAssembly();
    }

    if (vector_view && vector_view != matrix_view) {
        vector_view->endAssemblyPhase();
        vector_view->finalizeAssembly();
    }
}

void ParallelAssembler::reset()
{
    local_assembler_.reset();
    ghost_manager_.clearSendBuffers();
    ghost_manager_.clearReceivedContributions();
    row_dofs_.clear();
    col_dofs_.clear();
    owned_contributions_.clear();
    initialized_ = false;
}

// ============================================================================
// Matrix Assembly
// ============================================================================

AssemblyResult ParallelAssembler::assembleMatrix(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view)
{
    return assembleCellsParallel(mesh, test_space, trial_space, kernel,
                                 &matrix_view, nullptr, true, false);
}

// ============================================================================
// Vector Assembly
// ============================================================================

AssemblyResult ParallelAssembler::assembleVector(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView& vector_view)
{
    return assembleCellsParallel(mesh, space, space, kernel,
                                 nullptr, &vector_view, false, true);
}

// ============================================================================
// Combined Assembly
// ============================================================================

AssemblyResult ParallelAssembler::assembleBoth(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView& vector_view)
{
    return assembleCellsParallel(mesh, test_space, trial_space, kernel,
                                 &matrix_view, &vector_view, true, true);
}

// ============================================================================
// Face Assembly
// ============================================================================

AssemblyResult ParallelAssembler::assembleBoundaryFaces(
    const IMeshAccess& mesh,
    int boundary_marker,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    // Delegate to local assembler - boundary faces are typically owned locally
    return local_assembler_.assembleBoundaryFaces(
        mesh, boundary_marker, space, kernel, matrix_view, vector_view);
}

AssemblyResult ParallelAssembler::assembleInteriorFaces(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView* vector_view)
{
    // Interior face assembly is more complex in parallel
    // For now, delegate to local assembler
    // Full implementation would handle shared faces specially
    return local_assembler_.assembleInteriorFaces(
        mesh, test_space, trial_space, kernel, matrix_view, vector_view);
}

// ============================================================================
// Internal Implementation
// ============================================================================

AssemblyResult ParallelAssembler::assembleCellsParallel(
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
    auto start_time = std::chrono::steady_clock::now();

    if (!initialized_) {
        initialize();
    }

    // Clear ghost buffers from previous assembly
    ghost_manager_.clearSendBuffers();

    // Begin assembly phases
    if (matrix_view && assemble_matrix) {
        matrix_view->beginAssemblyPhase();
    }
    if (vector_view && assemble_vector && vector_view != matrix_view) {
        vector_view->beginAssemblyPhase();
    }

    const auto required_data = kernel.getRequiredData();

    // Iterate over OWNED cells only for parallel assembly
    mesh.forEachOwnedCell([&](GlobalIndex cell_id) {
        // Get element DOFs
        auto test_dofs = dof_map_->getCellDofs(cell_id);
        row_dofs_.assign(test_dofs.begin(), test_dofs.end());
        col_dofs_.assign(test_dofs.begin(), test_dofs.end());

        // Get cell type
        ElementType cell_type = mesh.getCellType(cell_id);

        // Prepare context
        const auto& test_element = test_space.getElement(cell_type, cell_id);
        const auto& trial_element = trial_space.getElement(cell_type, cell_id);
        context_.configure(cell_id, test_element, trial_element, required_data);

        // Compute local matrix/vector
        kernel_output_.clear();
        kernel.computeCell(context_, kernel_output_);

        // Insert with ghost handling
        insertLocalWithGhostHandling(
            kernel_output_, row_dofs_, col_dofs_,
            assemble_matrix ? matrix_view : nullptr,
            assemble_vector ? vector_view : nullptr);

        result.elements_assembled++;

        if (kernel_output_.has_matrix) {
            result.matrix_entries_inserted +=
                static_cast<GlobalIndex>(row_dofs_.size() * col_dofs_.size());
        }
        if (kernel_output_.has_vector) {
            result.vector_entries_inserted +=
                static_cast<GlobalIndex>(row_dofs_.size());
        }
    });

    // Also assemble ghost cells (contributions to owned DOFs)
    // This is important for ReverseScatter policy
    if (ghost_policy_ == GhostPolicy::ReverseScatter) {
        mesh.forEachCell([&](GlobalIndex cell_id) {
            // Skip owned cells (already processed)
            if (mesh.isOwnedCell(cell_id)) return;

            auto test_dofs = dof_map_->getCellDofs(cell_id);
            row_dofs_.assign(test_dofs.begin(), test_dofs.end());
            col_dofs_.assign(test_dofs.begin(), test_dofs.end());

            ElementType cell_type = mesh.getCellType(cell_id);

            const auto& test_element = test_space.getElement(cell_type, cell_id);
            const auto& trial_element = trial_space.getElement(cell_type, cell_id);
            context_.configure(cell_id, test_element, trial_element, required_data);

            kernel_output_.clear();
            kernel.computeCell(context_, kernel_output_);

            insertLocalWithGhostHandling(
                kernel_output_, row_dofs_, col_dofs_,
                assemble_matrix ? matrix_view : nullptr,
                assemble_vector ? vector_view : nullptr);
        });
    }

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

void ParallelAssembler::insertLocalWithGhostHandling(
    const KernelOutput& output,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    if (ghost_policy_ == GhostPolicy::OwnedRowsOnly) {
        // Only insert to owned rows
        const auto n_rows = static_cast<GlobalIndex>(row_dofs.size());
        const auto n_cols = static_cast<GlobalIndex>(col_dofs.size());

        for (GlobalIndex i = 0; i < n_rows; ++i) {
            GlobalIndex row = row_dofs[static_cast<std::size_t>(i)];

            if (!ghost_manager_.isOwned(row)) continue;

            // Matrix row
            if (matrix_view && output.has_matrix) {
                for (GlobalIndex j = 0; j < n_cols; ++j) {
                    GlobalIndex col = col_dofs[static_cast<std::size_t>(j)];
                    Real val = output.matrixEntry(i, j);
                    matrix_view->addMatrixEntry(row, col, val);
                }
            }

            // Vector entry
            if (vector_view && output.has_vector) {
                Real val = output.vectorEntry(i);
                vector_view->addVectorEntry(row, val);
            }
        }
    } else {
        // ReverseScatter: insert all, buffering ghosts
        const auto n_rows = static_cast<GlobalIndex>(row_dofs.size());
        const auto n_cols = static_cast<GlobalIndex>(col_dofs.size());

        for (GlobalIndex i = 0; i < n_rows; ++i) {
            GlobalIndex row = row_dofs[static_cast<std::size_t>(i)];
            bool is_owned = ghost_manager_.isOwned(row);

            // Matrix entries
            if (output.has_matrix) {
                for (GlobalIndex j = 0; j < n_cols; ++j) {
                    GlobalIndex col = col_dofs[static_cast<std::size_t>(j)];
                    Real val = output.matrixEntry(i, j);

                    if (is_owned) {
                        if (matrix_view) {
                            matrix_view->addMatrixEntry(row, col, val);
                        }
                    } else {
                        ghost_manager_.addMatrixContribution(row, col, val);
                    }
                }
            }

            // Vector entry
            if (output.has_vector) {
                Real val = output.vectorEntry(i);

                if (is_owned) {
                    if (vector_view) {
                        vector_view->addVectorEntry(row, val);
                    }
                } else {
                    ghost_manager_.addVectorContribution(row, val);
                }
            }
        }
    }
}

void ParallelAssembler::exchangeGhostContributions()
{
    ghost_manager_.exchangeContributions();
}

void ParallelAssembler::applyReceivedContributions(
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    // Apply received matrix contributions
    if (matrix_view) {
        auto received = ghost_manager_.getReceivedMatrixContributions();
        for (const auto& entry : received) {
            matrix_view->addMatrixEntry(entry.global_row, entry.global_col, entry.value);
        }
    }

    // Apply received vector contributions
    if (vector_view) {
        auto received = ghost_manager_.getReceivedVectorContributions();
        for (const auto& entry : received) {
            vector_view->addVectorEntry(entry.first, entry.second);
        }
    }

    ghost_manager_.clearReceivedContributions();
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<Assembler> createParallelAssembler()
{
    return std::make_unique<ParallelAssembler>();
}

#if FE_HAS_MPI
std::unique_ptr<Assembler> createParallelAssembler(MPI_Comm comm)
{
    return std::make_unique<ParallelAssembler>(comm);
}
#endif

std::unique_ptr<Assembler> createParallelAssembler(const AssemblyOptions& options)
{
    return std::make_unique<ParallelAssembler>(options);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
