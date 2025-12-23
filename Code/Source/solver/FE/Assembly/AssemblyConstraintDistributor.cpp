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

#include "AssemblyConstraintDistributor.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/ConstraintDistributor.h"
#include "Core/FEException.h"

#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// GlobalSystemMatrixAdapter Implementation
// ============================================================================

GlobalSystemMatrixAdapter::GlobalSystemMatrixAdapter(GlobalSystemView& view)
    : view_(view)
{
}

void GlobalSystemMatrixAdapter::addValues(
    std::span<const GlobalIndex> rows,
    std::span<const GlobalIndex> cols,
    std::span<const double> values)
{
    // Convert from row-major flat array to GlobalSystemView format
    view_.addMatrixEntries(rows, cols, values);
}

void GlobalSystemMatrixAdapter::addValue(GlobalIndex row, GlobalIndex col, double value) {
    view_.addMatrixEntry(row, col, static_cast<Real>(value));
}

void GlobalSystemMatrixAdapter::setDiagonal(GlobalIndex row, double value) {
    view_.setDiagonal(row, static_cast<Real>(value));
}

GlobalIndex GlobalSystemMatrixAdapter::numRows() const {
    return view_.numRows();
}

GlobalIndex GlobalSystemMatrixAdapter::numCols() const {
    return view_.numCols();
}

// ============================================================================
// GlobalSystemVectorAdapter Implementation
// ============================================================================

GlobalSystemVectorAdapter::GlobalSystemVectorAdapter(GlobalSystemView& view)
    : view_(view)
{
}

void GlobalSystemVectorAdapter::addValues(
    std::span<const GlobalIndex> indices,
    std::span<const double> values)
{
    view_.addVectorEntries(indices, values);
}

void GlobalSystemVectorAdapter::addValue(GlobalIndex index, double value) {
    view_.addVectorEntry(index, static_cast<Real>(value));
}

void GlobalSystemVectorAdapter::setValue(GlobalIndex index, double value) {
    std::array<GlobalIndex, 1> idx = {index};
    std::array<Real, 1> val = {static_cast<Real>(value)};
    view_.setVectorEntries(idx, val);
}

double GlobalSystemVectorAdapter::getValue(GlobalIndex index) const {
    return static_cast<double>(view_.getVectorEntry(index));
}

GlobalIndex GlobalSystemVectorAdapter::size() const {
    return view_.numRows();
}

// ============================================================================
// AssemblyConstraintDistributor Implementation
// ============================================================================

AssemblyConstraintDistributor::AssemblyConstraintDistributor()
    : constraints_(nullptr)
    , options_{}
{
}

AssemblyConstraintDistributor::AssemblyConstraintDistributor(
    const constraints::AffineConstraints& constraints)
    : constraints_(&constraints)
    , options_{}
{
    FE_THROW_IF(!constraints.isClosed(),
               "Constraints must be closed before use in AssemblyConstraintDistributor");

    // Create the underlying distributor from Constraints module
    distributor_ = std::make_unique<constraints::ConstraintDistributor>(constraints);
}

AssemblyConstraintDistributor::AssemblyConstraintDistributor(
    const constraints::AffineConstraints& constraints,
    const AssemblyConstraintOptions& options)
    : constraints_(&constraints)
    , options_(options)
{
    FE_THROW_IF(!constraints.isClosed(),
               "Constraints must be closed before use in AssemblyConstraintDistributor");

    // Create and configure the underlying distributor
    constraints::DistributorOptions dist_options;
    dist_options.symmetric = options.symmetric_elimination;
    dist_options.constrained_diagonal = options.constrained_diagonal;
    dist_options.zero_tolerance = options.zero_tolerance;
    dist_options.apply_inhomogeneities = options.apply_inhomogeneities;

    distributor_ = std::make_unique<constraints::ConstraintDistributor>(constraints, dist_options);
}

AssemblyConstraintDistributor::~AssemblyConstraintDistributor() = default;

AssemblyConstraintDistributor::AssemblyConstraintDistributor(
    AssemblyConstraintDistributor&& other) noexcept = default;

AssemblyConstraintDistributor& AssemblyConstraintDistributor::operator=(
    AssemblyConstraintDistributor&& other) noexcept = default;

void AssemblyConstraintDistributor::setConstraints(
    const constraints::AffineConstraints& constraints)
{
    FE_THROW_IF(!constraints.isClosed(),
               "Constraints must be closed before use in AssemblyConstraintDistributor");

    constraints_ = &constraints;

    // Recreate distributor with current options
    constraints::DistributorOptions dist_options;
    dist_options.symmetric = options_.symmetric_elimination;
    dist_options.constrained_diagonal = options_.constrained_diagonal;
    dist_options.zero_tolerance = options_.zero_tolerance;
    dist_options.apply_inhomogeneities = options_.apply_inhomogeneities;

    distributor_ = std::make_unique<constraints::ConstraintDistributor>(constraints, dist_options);
}

void AssemblyConstraintDistributor::setOptions(const AssemblyConstraintOptions& options) {
    options_ = options;

    // Update underlying distributor options if it exists
    if (distributor_) {
        constraints::DistributorOptions dist_options;
        dist_options.symmetric = options.symmetric_elimination;
        dist_options.constrained_diagonal = options.constrained_diagonal;
        dist_options.zero_tolerance = options.zero_tolerance;
        dist_options.apply_inhomogeneities = options.apply_inhomogeneities;
        distributor_->setOptions(dist_options);
    }
}

// ============================================================================
// Element Distribution
// ============================================================================

void AssemblyConstraintDistributor::distributeLocalToGlobal(
    std::span<const Real> cell_matrix,
    std::span<const Real> cell_rhs,
    std::span<const GlobalIndex> cell_dofs,
    GlobalSystemView& matrix_view,
    GlobalSystemView& rhs_view)
{
    distributeLocalToGlobal(cell_matrix, cell_rhs, cell_dofs, cell_dofs,
                            matrix_view, rhs_view);
}

void AssemblyConstraintDistributor::distributeMatrixToGlobal(
    std::span<const Real> cell_matrix,
    std::span<const GlobalIndex> cell_dofs,
    GlobalSystemView& matrix_view)
{
    if (!options_.apply_constraints || !hasConstraints()) {
        // Direct insertion without constraint handling
        matrix_view.addMatrixEntries(cell_dofs, cell_dofs, cell_matrix);
        return;
    }

    // Optimization: skip if no DOFs are constrained
    if (options_.skip_unconstrained && !hasConstrainedDofs(cell_dofs)) {
        matrix_view.addMatrixEntries(cell_dofs, cell_dofs, cell_matrix);
        return;
    }

    // Create adapter and use underlying distributor
    GlobalSystemMatrixAdapter matrix_adapter(matrix_view);
    distributor_->distributeMatrixToGlobal(cell_matrix, cell_dofs, matrix_adapter);
}

void AssemblyConstraintDistributor::distributeVectorToGlobal(
    std::span<const Real> cell_rhs,
    std::span<const GlobalIndex> cell_dofs,
    GlobalSystemView& rhs_view)
{
    if (!options_.apply_constraints || !hasConstraints()) {
        // Direct insertion without constraint handling
        rhs_view.addVectorEntries(cell_dofs, cell_rhs);
        return;
    }

    // Optimization: skip if no DOFs are constrained
    if (options_.skip_unconstrained && !hasConstrainedDofs(cell_dofs)) {
        rhs_view.addVectorEntries(cell_dofs, cell_rhs);
        return;
    }

    // Create adapter and use underlying distributor
    GlobalSystemVectorAdapter rhs_adapter(rhs_view);
    distributor_->distributeRhsToGlobal(cell_rhs, cell_dofs, rhs_adapter);
}

void AssemblyConstraintDistributor::distributeLocalToGlobal(
    std::span<const Real> cell_matrix,
    std::span<const Real> cell_rhs,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    GlobalSystemView& matrix_view,
    GlobalSystemView& rhs_view)
{
    if (!options_.apply_constraints || !hasConstraints()) {
        // Direct insertion
        distributeDirectly(cell_matrix, cell_rhs, row_dofs, col_dofs,
                          matrix_view, rhs_view);
        return;
    }

    // Optimization: skip if no DOFs are constrained
    if (options_.skip_unconstrained) {
        bool has_row_constraints = hasConstrainedDofs(row_dofs);
        bool has_col_constraints = hasConstrainedDofs(col_dofs);

        if (!has_row_constraints && !has_col_constraints) {
            distributeDirectly(cell_matrix, cell_rhs, row_dofs, col_dofs,
                              matrix_view, rhs_view);
            return;
        }
    }

    // Create adapters
    GlobalSystemMatrixAdapter matrix_adapter(matrix_view);
    GlobalSystemVectorAdapter rhs_adapter(rhs_view);

    // For square systems, use the standard distribution
    if (row_dofs.data() == col_dofs.data() && row_dofs.size() == col_dofs.size()) {
        distributor_->distributeLocalToGlobal(cell_matrix, cell_rhs, row_dofs,
                                              matrix_adapter, rhs_adapter);
    } else {
        // Rectangular systems - handle row and column constraints separately
        distributeCore(cell_matrix, cell_rhs, row_dofs, col_dofs,
                      &matrix_adapter, &rhs_adapter);
    }
}

void AssemblyConstraintDistributor::distributeCore(
    std::span<const Real> cell_matrix,
    std::span<const Real> cell_rhs,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    GlobalSystemMatrixAdapter* matrix_adapter,
    GlobalSystemVectorAdapter* rhs_adapter)
{
    const auto n_rows = static_cast<LocalIndex>(row_dofs.size());
    const auto n_cols = static_cast<LocalIndex>(col_dofs.size());

    // Allocate scratch space
    scratch_rows_.clear();
    scratch_cols_.clear();
    scratch_matrix_.clear();
    scratch_rhs_.clear();

    // Process each local entry
    for (LocalIndex i = 0; i < n_rows; ++i) {
        GlobalIndex global_row = row_dofs[i];

        // Check if row DOF is constrained
        auto row_constraint = constraints_->getConstraint(global_row);

        if (!row_constraint) {
            // Unconstrained row - distribute normally

            // RHS contribution
            if (rhs_adapter && !cell_rhs.empty()) {
                Real rhs_value = cell_rhs[i];

                // If symmetric elimination and column DOFs can be constrained,
                // need to add inhomogeneity contribution
                if (options_.symmetric_elimination && options_.apply_inhomogeneities) {
                    for (LocalIndex j = 0; j < n_cols; ++j) {
                        GlobalIndex global_col = col_dofs[j];
                        auto col_constraint = constraints_->getConstraint(global_col);
                        if (col_constraint && !col_constraint->isHomogeneous()) {
                            Real mat_entry = cell_matrix[static_cast<std::size_t>(i * n_cols + j)];
                            rhs_value -= mat_entry * col_constraint->inhomogeneity;
                        }
                    }
                }

                if (std::abs(rhs_value) > options_.zero_tolerance) {
                    rhs_adapter->addValue(global_row, rhs_value);
                }
            }

            // Matrix row contribution
            if (matrix_adapter && !cell_matrix.empty()) {
                for (LocalIndex j = 0; j < n_cols; ++j) {
                    GlobalIndex global_col = col_dofs[j];
                    Real mat_entry = cell_matrix[static_cast<std::size_t>(i * n_cols + j)];

                    if (std::abs(mat_entry) < options_.zero_tolerance) {
                        continue;
                    }

                    auto col_constraint = constraints_->getConstraint(global_col);

                    if (!col_constraint) {
                        // Unconstrained column
                        matrix_adapter->addValue(global_row, global_col, mat_entry);
                    } else {
                        // Constrained column - distribute to master DOFs
                        for (const auto& entry : col_constraint->entries) {
                            matrix_adapter->addValue(global_row, entry.master_dof,
                                                    mat_entry * entry.weight);
                        }
                    }
                }
            }
        } else {
            // Constrained row - distribute to master DOFs

            // Distribute RHS to masters
            if (rhs_adapter && !cell_rhs.empty()) {
                Real rhs_value = cell_rhs[i];

                for (const auto& row_entry : row_constraint->entries) {
                    Real contrib = rhs_value * row_entry.weight;
                    if (std::abs(contrib) > options_.zero_tolerance) {
                        rhs_adapter->addValue(row_entry.master_dof, contrib);
                    }
                }
            }

            // Distribute matrix to master DOFs
            if (matrix_adapter && !cell_matrix.empty()) {
                for (const auto& row_entry : row_constraint->entries) {
                    GlobalIndex master_row = row_entry.master_dof;
                    Real row_weight = row_entry.weight;

                    for (LocalIndex j = 0; j < n_cols; ++j) {
                        GlobalIndex global_col = col_dofs[j];
                        Real mat_entry = cell_matrix[static_cast<std::size_t>(i * n_cols + j)];

                        if (std::abs(mat_entry) < options_.zero_tolerance) {
                            continue;
                        }

                        auto col_constraint = constraints_->getConstraint(global_col);

                        if (!col_constraint) {
                            // Unconstrained column
                            matrix_adapter->addValue(master_row, global_col,
                                                    mat_entry * row_weight);
                        } else {
                            // Both row and column constrained
                            for (const auto& col_entry : col_constraint->entries) {
                                Real contrib = mat_entry * row_weight * col_entry.weight;
                                if (std::abs(contrib) > options_.zero_tolerance) {
                                    matrix_adapter->addValue(master_row, col_entry.master_dof,
                                                            contrib);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void AssemblyConstraintDistributor::distributeDirectly(
    std::span<const Real> cell_matrix,
    std::span<const Real> cell_rhs,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    GlobalSystemView& matrix_view,
    GlobalSystemView& rhs_view)
{
    if (!cell_matrix.empty()) {
        matrix_view.addMatrixEntries(row_dofs, col_dofs, cell_matrix);
    }
    if (!cell_rhs.empty()) {
        rhs_view.addVectorEntries(row_dofs, cell_rhs);
    }
}

// ============================================================================
// Finalization
// ============================================================================

void AssemblyConstraintDistributor::finalizeConstrainedRows(GlobalSystemView& matrix_view) {
    if (!hasConstraints()) {
        return;
    }

    auto constrained_dofs = getConstrainedDofs();
    for (GlobalIndex dof : constrained_dofs) {
        matrix_view.setDiagonal(dof, static_cast<Real>(options_.constrained_diagonal));
    }
}

void AssemblyConstraintDistributor::setConstrainedRhsValues(GlobalSystemView& rhs_view) {
    if (!hasConstraints()) {
        return;
    }

    constraints_->forEach([&rhs_view](const constraints::AffineConstraints::ConstraintView& cv) {
        std::array<GlobalIndex, 1> idx = {cv.slave_dof};
        std::array<Real, 1> val = {static_cast<Real>(cv.inhomogeneity)};
        rhs_view.setVectorEntries(idx, val);
    });
}

void AssemblyConstraintDistributor::zeroConstrainedEntries(GlobalSystemView& vec_view) {
    if (!hasConstraints()) {
        return;
    }

    auto constrained_dofs = getConstrainedDofs();
    vec_view.zeroVectorEntries(constrained_dofs);
}

// ============================================================================
// Solution Post-Processing
// ============================================================================

void AssemblyConstraintDistributor::distributeSolution(GlobalSystemView& solution_view) {
    if (!hasConstraints()) {
        return;
    }

    // Get solution size
    GlobalIndex size = solution_view.numRows();

    // Create temporary solution array
    std::vector<Real> solution(static_cast<std::size_t>(size));

    // Extract current values
    for (GlobalIndex i = 0; i < size; ++i) {
        solution[static_cast<std::size_t>(i)] = solution_view.getVectorEntry(i);
    }

    // Apply constraints
    constraints_->distribute(solution);

    // Write back
    for (GlobalIndex i = 0; i < size; ++i) {
        std::array<GlobalIndex, 1> idx = {i};
        std::array<Real, 1> val = {solution[static_cast<std::size_t>(i)]};
        solution_view.setVectorEntries(idx, val);
    }
}

void AssemblyConstraintDistributor::distributeSolution(Real* solution, GlobalIndex size) {
    if (!hasConstraints()) {
        return;
    }

    // Convert Real* to double* if needed and call constraints distribute
    std::vector<double> temp(static_cast<std::size_t>(size));
    for (GlobalIndex i = 0; i < size; ++i) {
        temp[static_cast<std::size_t>(i)] = static_cast<double>(solution[i]);
    }

    constraints_->distribute(temp);

    for (GlobalIndex i = 0; i < size; ++i) {
        solution[i] = static_cast<Real>(temp[static_cast<std::size_t>(i)]);
    }
}

// ============================================================================
// Query
// ============================================================================

bool AssemblyConstraintDistributor::hasConstrainedDofs(
    std::span<const GlobalIndex> dofs) const
{
    if (!hasConstraints()) {
        return false;
    }
    return constraints_->hasConstrainedDofs(dofs);
}

bool AssemblyConstraintDistributor::isConstrained(GlobalIndex dof) const {
    if (!hasConstraints()) {
        return false;
    }
    return constraints_->isConstrained(dof);
}

std::vector<GlobalIndex> AssemblyConstraintDistributor::getConstrainedDofs() const {
    if (!hasConstraints()) {
        return {};
    }
    return constraints_->getConstrainedDofs();
}

std::size_t AssemblyConstraintDistributor::numConstraints() const {
    if (!hasConstraints()) {
        return 0;
    }
    return constraints_->numConstraints();
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<AssemblyConstraintDistributor> createAssemblyConstraintDistributor(
    const constraints::AffineConstraints& constraints)
{
    return std::make_unique<AssemblyConstraintDistributor>(constraints);
}

std::unique_ptr<AssemblyConstraintDistributor> createAssemblyConstraintDistributor(
    const constraints::AffineConstraints& constraints,
    const AssemblyConstraintOptions& options)
{
    return std::make_unique<AssemblyConstraintDistributor>(constraints, options);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
