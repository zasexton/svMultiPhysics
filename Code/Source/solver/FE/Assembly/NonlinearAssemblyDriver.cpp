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

#include "NonlinearAssemblyDriver.h"
#include "AssemblyConstraintDistributor.h"
#include "Constraints/AffineConstraints.h"
#include "Core/FEException.h"
#include "Dofs/DofMap.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <utility>

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// NonlinearAssemblyDriver Implementation
// ============================================================================

NonlinearAssemblyDriver::NonlinearAssemblyDriver()
    : options_{}
    , loop_(std::make_unique<AssemblyLoop>())
{
}

NonlinearAssemblyDriver::NonlinearAssemblyDriver(const NonlinearAssemblyOptions& options)
    : options_(options)
    , loop_(std::make_unique<AssemblyLoop>(options.loop_options))
{
}

NonlinearAssemblyDriver::~NonlinearAssemblyDriver() = default;

NonlinearAssemblyDriver::NonlinearAssemblyDriver(NonlinearAssemblyDriver&& other) noexcept = default;

NonlinearAssemblyDriver& NonlinearAssemblyDriver::operator=(
    NonlinearAssemblyDriver&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void NonlinearAssemblyDriver::setMesh(const IMeshAccess& mesh) {
    mesh_ = &mesh;
    loop_->setMesh(mesh);
}

void NonlinearAssemblyDriver::setDofMap(const dofs::DofMap& dof_map) {
    dof_map_ = &dof_map;
    loop_->setDofMap(dof_map);
}

void NonlinearAssemblyDriver::setSpace(const spaces::FunctionSpace& space) {
    test_space_ = &space;
    trial_space_ = &space;
}

void NonlinearAssemblyDriver::setSpaces(
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space)
{
    test_space_ = &test_space;
    trial_space_ = &trial_space;
}

void NonlinearAssemblyDriver::setKernel(INonlinearKernel& kernel) {
    kernel_ = &kernel;
}

void NonlinearAssemblyDriver::setConstraints(const constraints::AffineConstraints& constraints) {
    constraints_ = &constraints;
    constraint_distributor_ = std::make_unique<AssemblyConstraintDistributor>(constraints);
}

void NonlinearAssemblyDriver::setOptions(const NonlinearAssemblyOptions& options) {
    options_ = options;
    loop_->setOptions(options.loop_options);
}

bool NonlinearAssemblyDriver::isConfigured() const noexcept {
    return mesh_ != nullptr &&
           dof_map_ != nullptr &&
           test_space_ != nullptr &&
           trial_space_ != nullptr &&
           kernel_ != nullptr;
}

// ============================================================================
// Solution Management
// ============================================================================

void NonlinearAssemblyDriver::setCurrentSolution(std::span<const Real> solution) {
    current_solution_.assign(solution.begin(), solution.end());
}

void NonlinearAssemblyDriver::setCurrentSolution(const GlobalSystemView& solution_view) {
    GlobalIndex size = solution_view.numRows();
    current_solution_.resize(static_cast<std::size_t>(size));

    for (GlobalIndex i = 0; i < size; ++i) {
        current_solution_[static_cast<std::size_t>(i)] = solution_view.getVectorEntry(i);
    }
}

// ============================================================================
// Assembly Operations
// ============================================================================

NonlinearAssemblyStats NonlinearAssemblyDriver::assembleResidual(GlobalSystemView& residual_view) {
    FE_THROW_IF(!isConfigured(), "NonlinearAssemblyDriver not configured");
    FE_THROW_IF(!kernel_->canComputeResidual(), "Kernel cannot compute residual");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = NonlinearAssemblyStats{};

    // Zero residual if requested
    if (options_.zero_before_assembly) {
        residual_view.zero();
    }

    // Cell loop
    residualCellLoop(residual_view);

    // Boundary loop
    if (options_.include_boundary && !options_.boundary_markers.empty()) {
        residualBoundaryLoop(residual_view);
    }

    // Apply constraints
    if (options_.apply_constraints && constraints_ != nullptr) {
        applyConstraints(nullptr, &residual_view);
    }

    // Finalize assembly
    residual_view.endAssemblyPhase();

    // Compute residual norm
    last_stats_.residual_norm = computeResidualNorm(residual_view);

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.residual_assembly_seconds =
        std::chrono::duration<double>(end_time - start_time).count();
    last_stats_.total_seconds = last_stats_.residual_assembly_seconds;

    return last_stats_;
}

NonlinearAssemblyStats NonlinearAssemblyDriver::assembleJacobian(GlobalSystemView& jacobian_view) {
    FE_THROW_IF(!isConfigured(), "NonlinearAssemblyDriver not configured");
    FE_THROW_IF(!kernel_->canComputeJacobian(), "Kernel cannot compute Jacobian");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = NonlinearAssemblyStats{};

    // Zero Jacobian if requested
    if (options_.zero_before_assembly) {
        jacobian_view.zero();
    }

    // Use FD if requested
    if (options_.jacobian_strategy == JacobianStrategy::FiniteDifference) {
        return assembleJacobianFD(jacobian_view);
    }

    // Cell loop
    jacobianCellLoop(jacobian_view);

    // Boundary loop
    if (options_.include_boundary && !options_.boundary_markers.empty()) {
        jacobianBoundaryLoop(jacobian_view);
    }

    // Apply constraints
    if (options_.apply_constraints && constraints_ != nullptr) {
        applyConstraints(&jacobian_view, nullptr);
    }

    // Finalize assembly
    jacobian_view.endAssemblyPhase();

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.jacobian_assembly_seconds =
        std::chrono::duration<double>(end_time - start_time).count();
    last_stats_.total_seconds = last_stats_.jacobian_assembly_seconds;

    return last_stats_;
}

NonlinearAssemblyStats NonlinearAssemblyDriver::assembleBoth(
    GlobalSystemView& jacobian_view,
    GlobalSystemView& residual_view)
{
    FE_THROW_IF(!isConfigured(), "NonlinearAssemblyDriver not configured");
    FE_THROW_IF(!kernel_->canComputeResidual(), "Kernel cannot compute residual");
    FE_THROW_IF(!kernel_->canComputeJacobian(), "Kernel cannot compute Jacobian");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = NonlinearAssemblyStats{};

    // Zero if requested
    if (options_.zero_before_assembly) {
        jacobian_view.zero();
        residual_view.zero();
    }

    // Combined cell loop (most efficient)
    combinedCellLoop(jacobian_view, residual_view);

    // Apply constraints
    if (options_.apply_constraints && constraints_ != nullptr) {
        applyConstraints(&jacobian_view, &residual_view);
    }

    // Finalize assembly
    jacobian_view.endAssemblyPhase();
    residual_view.endAssemblyPhase();

    // Compute residual norm
    last_stats_.residual_norm = computeResidualNorm(residual_view);

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

// ============================================================================
// Cell Loops
// ============================================================================

void NonlinearAssemblyDriver::residualCellLoop(GlobalSystemView& residual_view) {
    residual_view.beginAssemblyPhase();

    RequiredData required = kernel_->getResidualRequiredData();

    loop_->cellLoop(
        *test_space_, *trial_space_, required,
        // Compute callback
        [this, required](const CellWorkItem& cell, AssemblyContext& context, KernelOutput& output) {
            prepareSolutionValues(context, cell.cell_id, required);
            output.clear();
            output.reserve(context.numTestDofs(), context.numTrialDofs(), false, true);
            kernel_->computeResidual(context, output);
        },
        // Insert callback
        [this, &residual_view](
            const CellWorkItem& /*cell*/,
            const KernelOutput& output,
            std::span<const GlobalIndex> row_dofs,
            std::span<const GlobalIndex> /*col_dofs*/) {
            if (constraint_distributor_ && options_.apply_constraints) {
                constraint_distributor_->distributeVectorToGlobal(
                    output.local_vector, row_dofs, residual_view);
            } else {
                residual_view.addVectorEntries(row_dofs, output.local_vector);
            }
        }
    );

    last_stats_.num_cells = loop_->getLastStatistics().total_iterations;
}

void NonlinearAssemblyDriver::jacobianCellLoop(GlobalSystemView& jacobian_view) {
    jacobian_view.beginAssemblyPhase();

    RequiredData required = kernel_->getJacobianRequiredData();

    loop_->cellLoop(
        *test_space_, *trial_space_, required,
        // Compute callback
        [this, required](const CellWorkItem& cell, AssemblyContext& context, KernelOutput& output) {
            prepareSolutionValues(context, cell.cell_id, required);
            output.clear();
            output.reserve(context.numTestDofs(), context.numTrialDofs(), true, false);
            kernel_->computeJacobian(context, output);
        },
        // Insert callback
        [this, &jacobian_view](
            const CellWorkItem& /*cell*/,
            const KernelOutput& output,
            std::span<const GlobalIndex> row_dofs,
            std::span<const GlobalIndex> col_dofs) {
            if (constraint_distributor_ && options_.apply_constraints) {
                constraint_distributor_->distributeMatrixToGlobal(
                    output.local_matrix, row_dofs, jacobian_view);
            } else {
                jacobian_view.addMatrixEntries(row_dofs, col_dofs, output.local_matrix);
            }
        }
    );

    last_stats_.num_cells = loop_->getLastStatistics().total_iterations;
}

void NonlinearAssemblyDriver::combinedCellLoop(
    GlobalSystemView& jacobian_view,
    GlobalSystemView& residual_view)
{
    jacobian_view.beginAssemblyPhase();
    residual_view.beginAssemblyPhase();

    // Use Jacobian required data (typically superset of residual data)
    RequiredData required = kernel_->getJacobianRequiredData();

    loop_->cellLoop(
        *test_space_, *trial_space_, required,
        // Compute callback
        [this, required](const CellWorkItem& cell, AssemblyContext& context, KernelOutput& output) {
            prepareSolutionValues(context, cell.cell_id, required);
            output.clear();
            output.reserve(context.numTestDofs(), context.numTrialDofs(), true, true);

            if (kernel_->hasOptimizedBoth()) {
                kernel_->computeBoth(context, output);
            } else {
                kernel_->computeResidual(context, output);
                kernel_->computeJacobian(context, output);
            }
        },
        // Insert callback
        [this, &jacobian_view, &residual_view](
            const CellWorkItem& /*cell*/,
            const KernelOutput& output,
            std::span<const GlobalIndex> row_dofs,
            std::span<const GlobalIndex> col_dofs) {
            if (constraint_distributor_ && options_.apply_constraints) {
                constraint_distributor_->distributeLocalToGlobal(
                    output.local_matrix, output.local_vector,
                    row_dofs, col_dofs, jacobian_view, residual_view);
            } else {
                jacobian_view.addMatrixEntries(row_dofs, col_dofs, output.local_matrix);
                residual_view.addVectorEntries(row_dofs, output.local_vector);
            }
        }
    );

    last_stats_.num_cells = loop_->getLastStatistics().total_iterations;
}

void NonlinearAssemblyDriver::residualBoundaryLoop(GlobalSystemView& /*residual_view*/) {
    // Boundary loop implementation would go here
    // Similar to residualCellLoop but iterating over boundary faces
    // and calling kernel boundary methods

    // Placeholder - actual implementation depends on kernel interface
    last_stats_.num_boundary_faces = 0;
}

void NonlinearAssemblyDriver::jacobianBoundaryLoop(GlobalSystemView& /*jacobian_view*/) {
    // Boundary loop implementation
    last_stats_.num_boundary_faces = 0;
}

// ============================================================================
// Helpers
// ============================================================================

void NonlinearAssemblyDriver::prepareSolutionValues(
    AssemblyContext& context,
    GlobalIndex cell_id,
    RequiredData required_data)
{
    const bool need_solution =
        hasFlag(required_data, RequiredData::SolutionCoefficients) ||
        hasFlag(required_data, RequiredData::SolutionValues) ||
        hasFlag(required_data, RequiredData::SolutionGradients) ||
        hasFlag(required_data, RequiredData::SolutionHessians) ||
        hasFlag(required_data, RequiredData::SolutionLaplacians);

    if (!need_solution) {
        return;
    }

    FE_THROW_IF(dof_map_ == nullptr, "NonlinearAssemblyDriver::prepareSolutionValues: DofMap not configured");
    FE_THROW_IF(current_solution_.empty(), "NonlinearAssemblyDriver::prepareSolutionValues: current solution not set");

    const auto cell_dofs = dof_map_->getCellDofs(cell_id);
    const auto n_trial = context.numTrialDofs();
    FE_THROW_IF(cell_dofs.size() < static_cast<std::size_t>(n_trial),
                "NonlinearAssemblyDriver::prepareSolutionValues: cell DOF list is smaller than trial DOFs");

    thread_local std::vector<Real> local_coefficients;
    local_coefficients.resize(static_cast<std::size_t>(n_trial));

    for (LocalIndex j = 0; j < n_trial; ++j) {
        const GlobalIndex gdof = cell_dofs[static_cast<std::size_t>(j)];
        FE_THROW_IF(gdof < 0, "NonlinearAssemblyDriver::prepareSolutionValues: negative DOF id");
        FE_THROW_IF(static_cast<std::size_t>(gdof) >= current_solution_.size(),
                    "NonlinearAssemblyDriver::prepareSolutionValues: solution vector is too small for cell DOFs");
        local_coefficients[static_cast<std::size_t>(j)] = current_solution_[static_cast<std::size_t>(gdof)];
    }

    context.setSolutionCoefficients(local_coefficients);
}

void NonlinearAssemblyDriver::applyConstraints(
    GlobalSystemView* jacobian_view,
    GlobalSystemView* residual_view)
{
    if (!constraint_distributor_) {
        return;
    }

    // Finalize constrained rows in Jacobian
    if (jacobian_view) {
        constraint_distributor_->finalizeConstrainedRows(*jacobian_view);
    }

    // Set constrained RHS values
    if (residual_view) {
        constraint_distributor_->setConstrainedRhsValues(*residual_view);
    }
}

Real NonlinearAssemblyDriver::computeResidualNorm(const GlobalSystemView& residual_view) {
    GlobalIndex size = residual_view.numRows();
    Real sum_sq = 0.0;

    for (GlobalIndex i = 0; i < size; ++i) {
        Real val = residual_view.getVectorEntry(i);
        sum_sq += val * val;
    }

    return std::sqrt(sum_sq);
}

// ============================================================================
// Jacobian Verification
// ============================================================================

JacobianVerificationResult NonlinearAssemblyDriver::verifyJacobianFD(
    const GlobalSystemView& jacobian_view,
    Real tol)
{
    // Verify all DOFs
    GlobalIndex n = jacobian_view.numRows();
    std::vector<GlobalIndex> rows(static_cast<std::size_t>(n));
    for (GlobalIndex i = 0; i < n; ++i) {
        rows[static_cast<std::size_t>(i)] = i;
    }
    return verifyJacobianFD(jacobian_view, rows, tol);
}

JacobianVerificationResult NonlinearAssemblyDriver::verifyJacobianFD(
    const GlobalSystemView& jacobian_view,
    std::span<const GlobalIndex> rows,
    Real tol)
{
    JacobianVerificationResult result;

    FE_THROW_IF(!isConfigured(), "Driver not configured for verification");
    FE_THROW_IF(current_solution_.empty(), "Solution not set for verification");

    const GlobalIndex n = jacobian_view.numRows();
    FE_THROW_IF(n <= 0, "verifyJacobianFD: invalid Jacobian size");
    FE_THROW_IF(static_cast<std::size_t>(n) != current_solution_.size(),
                "verifyJacobianFD: solution size does not match Jacobian size");

    // Compute base residual F(u)
    residual_base_.assign(static_cast<std::size_t>(n), 0.0);
    {
        DenseVectorView base_residual_view(n);
        base_residual_view.zero();

        residualCellLoop(base_residual_view);
        if (options_.include_boundary && !options_.boundary_markers.empty()) {
            residualBoundaryLoop(base_residual_view);
        }
        if (options_.apply_constraints && constraints_ != nullptr) {
            applyConstraints(nullptr, &base_residual_view);
        }
        base_residual_view.endAssemblyPhase();

        const auto base = base_residual_view.data();
        FE_THROW_IF(base.size() != residual_base_.size(),
                    "verifyJacobianFD: base residual size mismatch");
        std::copy(base.begin(), base.end(), residual_base_.begin());
    }

    residual_perturbed_.assign(static_cast<std::size_t>(n), 0.0);
    perturbed_solution_ = current_solution_;

    Real max_abs_err = 0.0;
    Real max_rel_err = 0.0;
    Real sum_abs_err = 0.0;
    GlobalIndex worst_row = -1;
    GlobalIndex worst_col = -1;
    std::size_t num_checked = 0;

    // For each column (perturb each DOF)
    for (GlobalIndex j = 0; j < n; ++j) {
        // Compute perturbation
        Real u_j = current_solution_[static_cast<std::size_t>(j)];
        Real h = options_.fd_perturbation;
        if (options_.fd_relative_perturbation) {
            h *= (1.0 + std::abs(u_j));
        }

        // Perturb solution
        perturbed_solution_[static_cast<std::size_t>(j)] = u_j + h;

        // Compute FD column: (F(u + h*e_j) - F(u)) / h
        computeFDColumn(j, residual_perturbed_, h);

        // Compare with analytic Jacobian for specified rows
        for (GlobalIndex row : rows) {
            Real j_analytic = jacobian_view.getMatrixEntry(row, j);
            Real j_fd = residual_perturbed_[static_cast<std::size_t>(row)];

            Real abs_err = std::abs(j_analytic - j_fd);
            Real rel_err = abs_err / (std::max(std::abs(j_analytic), std::abs(j_fd)) + 1e-14);

            sum_abs_err += abs_err;
            ++num_checked;

            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
                max_rel_err = rel_err;
                worst_row = row;
                worst_col = j;
            }
        }

        // Restore solution
        perturbed_solution_[static_cast<std::size_t>(j)] = u_j;
    }

    // Build result
    result.max_abs_error = max_abs_err;
    result.max_rel_error = max_rel_err;
    result.avg_abs_error = num_checked > 0 ? sum_abs_err / static_cast<Real>(num_checked) : 0.0;
    result.worst_row = worst_row;
    result.worst_col = worst_col;
    result.num_entries_checked = num_checked;
    result.passed = (max_abs_err < tol) || (max_rel_err < tol);

    // Build message
    std::ostringstream oss;
    if (result.passed) {
        oss << "Jacobian verification PASSED. ";
    } else {
        oss << "Jacobian verification FAILED. ";
    }
    oss << "Max abs error: " << max_abs_err
        << ", Max rel error: " << max_rel_err
        << " at (" << worst_row << ", " << worst_col << ")";
    result.message = oss.str();

    return result;
}

NonlinearAssemblyStats NonlinearAssemblyDriver::assembleJacobianFD(
    GlobalSystemView& fd_jacobian_view)
{
    FE_THROW_IF(!isConfigured(), "Driver not configured for FD Jacobian");
    FE_THROW_IF(current_solution_.empty(), "Solution not set for FD Jacobian");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = NonlinearAssemblyStats{};

    const GlobalIndex n = fd_jacobian_view.numRows();
    FE_THROW_IF(n <= 0, "assembleJacobianFD: invalid Jacobian size");
    FE_THROW_IF(fd_jacobian_view.numCols() != n, "assembleJacobianFD: FD Jacobian view must be square");
    FE_THROW_IF(static_cast<std::size_t>(n) != current_solution_.size(),
                "assembleJacobianFD: solution size does not match Jacobian size");

    // Zero Jacobian
    fd_jacobian_view.zero();
    fd_jacobian_view.beginAssemblyPhase();

    perturbed_solution_ = current_solution_;

    // Base residual F(u)
    residual_base_.assign(static_cast<std::size_t>(n), 0.0);
    {
        DenseVectorView base_residual_view(n);
        base_residual_view.zero();

        residualCellLoop(base_residual_view);
        if (options_.include_boundary && !options_.boundary_markers.empty()) {
            residualBoundaryLoop(base_residual_view);
        }
        if (options_.apply_constraints && constraints_ != nullptr) {
            applyConstraints(nullptr, &base_residual_view);
        }
        base_residual_view.endAssemblyPhase();

        const auto base = base_residual_view.data();
        FE_THROW_IF(base.size() != residual_base_.size(),
                    "assembleJacobianFD: base residual size mismatch");
        std::copy(base.begin(), base.end(), residual_base_.begin());
    }

    // For each column
    for (GlobalIndex j = 0; j < n; ++j) {
        Real u_j = current_solution_[static_cast<std::size_t>(j)];
        Real h = options_.fd_perturbation;
        if (options_.fd_relative_perturbation) {
            h *= (1.0 + std::abs(u_j));
        }

        // Perturb
        perturbed_solution_[static_cast<std::size_t>(j)] = u_j + h;

        // Compute column
        std::vector<Real> column(static_cast<std::size_t>(n));
        computeFDColumn(j, column, h);

        // Insert column into matrix
        for (GlobalIndex i = 0; i < n; ++i) {
            if (std::abs(column[static_cast<std::size_t>(i)]) > options_.fd_perturbation * 1e-10) {
                fd_jacobian_view.addMatrixEntry(i, j, column[static_cast<std::size_t>(i)]);
            }
        }

        // Restore
        perturbed_solution_[static_cast<std::size_t>(j)] = u_j;
    }

    // Apply constraints if needed
    if (options_.apply_constraints && constraint_distributor_) {
        constraint_distributor_->finalizeConstrainedRows(fd_jacobian_view);
    }

    fd_jacobian_view.endAssemblyPhase();

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.jacobian_assembly_seconds =
        std::chrono::duration<double>(end_time - start_time).count();
    last_stats_.total_seconds = last_stats_.jacobian_assembly_seconds;

    return last_stats_;
}

void NonlinearAssemblyDriver::computeFDColumn(
    GlobalIndex /*col*/,
    std::vector<Real>& column,
    Real h)
{
    FE_THROW_IF(h == 0.0, "computeFDColumn: perturbation step is zero");
    FE_THROW_IF(residual_base_.size() != column.size(),
                "computeFDColumn: base residual not available (callers must compute F(u) first)");

    const GlobalIndex n = static_cast<GlobalIndex>(column.size());

    // Assemble residual at perturbed solution, then difference.
    DenseVectorView perturbed_residual_view(n);
    perturbed_residual_view.zero();

    std::swap(current_solution_, perturbed_solution_);

    residualCellLoop(perturbed_residual_view);
    if (options_.include_boundary && !options_.boundary_markers.empty()) {
        residualBoundaryLoop(perturbed_residual_view);
    }
    if (options_.apply_constraints && constraints_ != nullptr) {
        applyConstraints(nullptr, &perturbed_residual_view);
    }
    perturbed_residual_view.endAssemblyPhase();

    std::swap(current_solution_, perturbed_solution_);

    const auto pert = perturbed_residual_view.data();
    FE_THROW_IF(pert.size() != column.size(), "computeFDColumn: residual size mismatch");

    for (GlobalIndex i = 0; i < n; ++i) {
        column[static_cast<std::size_t>(i)] =
            (pert[static_cast<std::size_t>(i)] - residual_base_[static_cast<std::size_t>(i)]) / h;
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<NonlinearAssemblyDriver> createNonlinearAssemblyDriver(
    const NonlinearAssemblyOptions& options)
{
    return std::make_unique<NonlinearAssemblyDriver>(options);
}

NonlinearAssemblyStats assembleResidual(
    const IMeshAccess& mesh,
    const dofs::DofMap& dof_map,
    const spaces::FunctionSpace& space,
    INonlinearKernel& kernel,
    std::span<const Real> solution,
    GlobalSystemView& residual_view,
    const constraints::AffineConstraints* constraints)
{
    NonlinearAssemblyDriver driver;
    driver.setMesh(mesh);
    driver.setDofMap(dof_map);
    driver.setSpace(space);
    driver.setKernel(kernel);
    driver.setCurrentSolution(solution);

    if (constraints) {
        driver.setConstraints(*constraints);
    }

    return driver.assembleResidual(residual_view);
}

NonlinearAssemblyStats assembleBoth(
    const IMeshAccess& mesh,
    const dofs::DofMap& dof_map,
    const spaces::FunctionSpace& space,
    INonlinearKernel& kernel,
    std::span<const Real> solution,
    GlobalSystemView& jacobian_view,
    GlobalSystemView& residual_view,
    const constraints::AffineConstraints* constraints)
{
    NonlinearAssemblyDriver driver;
    driver.setMesh(mesh);
    driver.setDofMap(dof_map);
    driver.setSpace(space);
    driver.setKernel(kernel);
    driver.setCurrentSolution(solution);

    if (constraints) {
        driver.setConstraints(*constraints);
    }

    return driver.assembleBoth(jacobian_view, residual_view);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
