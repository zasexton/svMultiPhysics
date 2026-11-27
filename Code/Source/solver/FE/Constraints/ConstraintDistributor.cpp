/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "ConstraintDistributor.h"

#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// ConstraintDistributor implementation
// ============================================================================

ConstraintDistributor::ConstraintDistributor() = default;

ConstraintDistributor::ConstraintDistributor(const AffineConstraints& constraints)
    : constraints_(&constraints) {
    if (!constraints.isClosed()) {
        CONSTRAINT_THROW("ConstraintDistributor requires closed AffineConstraints");
    }
}

ConstraintDistributor::ConstraintDistributor(const AffineConstraints& constraints,
                                              const DistributorOptions& options)
    : constraints_(&constraints), options_(options) {
    if (!constraints.isClosed()) {
        CONSTRAINT_THROW("ConstraintDistributor requires closed AffineConstraints");
    }
}

ConstraintDistributor::~ConstraintDistributor() = default;

ConstraintDistributor::ConstraintDistributor(ConstraintDistributor&& other) noexcept = default;

ConstraintDistributor& ConstraintDistributor::operator=(ConstraintDistributor&& other) noexcept = default;

void ConstraintDistributor::setConstraints(const AffineConstraints& constraints) {
    if (!constraints.isClosed()) {
        CONSTRAINT_THROW("ConstraintDistributor requires closed AffineConstraints");
    }
    constraints_ = &constraints;
}

// ============================================================================
// Single element distribution
// ============================================================================

void ConstraintDistributor::distributeLocalToGlobal(
    std::span<const double> cell_matrix,
    std::span<const double> cell_rhs,
    std::span<const GlobalIndex> cell_dofs,
    IMatrixOperations& matrix,
    IVectorOperations& rhs) const
{
    if (!constraints_) {
        // No constraints - simple direct distribution
        matrix.addValues(cell_dofs, cell_dofs, cell_matrix);
        rhs.addValues(cell_dofs, cell_rhs);
        return;
    }

    distributeElementCore(cell_matrix, cell_rhs, cell_dofs, &matrix, &rhs);
}

void ConstraintDistributor::distributeLocalToGlobal(
    std::span<const double> cell_matrix,
    std::span<const double> cell_rhs,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    IMatrixOperations& matrix,
    IVectorOperations& rhs) const
{
    if (!constraints_) {
        matrix.addValues(row_dofs, col_dofs, cell_matrix);
        rhs.addValues(row_dofs, cell_rhs);
        return;
    }

    distributeElementCoreRectangular(cell_matrix, cell_rhs, row_dofs, col_dofs, &matrix, &rhs);
}

void ConstraintDistributor::distributeMatrixToGlobal(
    std::span<const double> cell_matrix,
    std::span<const GlobalIndex> cell_dofs,
    IMatrixOperations& matrix) const
{
    if (!constraints_) {
        matrix.addValues(cell_dofs, cell_dofs, cell_matrix);
        return;
    }

    std::span<const double> empty_rhs;
    distributeElementCore(cell_matrix, empty_rhs, cell_dofs, &matrix, nullptr);
}

void ConstraintDistributor::distributeMatrixToGlobal(
    std::span<const double> cell_matrix,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    IMatrixOperations& matrix) const
{
    if (!constraints_) {
        matrix.addValues(row_dofs, col_dofs, cell_matrix);
        return;
    }

    std::span<const double> empty_rhs;
    distributeElementCoreRectangular(cell_matrix, empty_rhs, row_dofs, col_dofs, &matrix, nullptr);
}

void ConstraintDistributor::distributeRhsToGlobal(
    std::span<const double> cell_rhs,
    std::span<const GlobalIndex> cell_dofs,
    IVectorOperations& rhs) const
{
    if (!constraints_) {
        rhs.addValues(cell_dofs, cell_rhs);
        return;
    }

    std::span<const double> empty_matrix;
    distributeElementCore(empty_matrix, cell_rhs, cell_dofs, nullptr, &rhs);
}

void ConstraintDistributor::distributeBlockContributions(
    std::span<const double> cell_matrix,
    std::span<const double> cell_rhs,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    IMatrixOperations& matrix,
    IVectorOperations& rhs) const
{
    distributeElementCoreRectangular(cell_matrix, cell_rhs, row_dofs, col_dofs,
                                     &matrix, &rhs, false);
}

void ConstraintDistributor::distributeElementCoreRectangular(
    std::span<const double> cell_matrix,
    std::span<const double> cell_rhs,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    IMatrixOperations* matrix,
    IVectorOperations* rhs,
    bool finalize_rows) const
{
    const std::size_t n_rows = row_dofs.size();
    const std::size_t n_cols = col_dofs.size();

    const bool same_dofs = n_rows == n_cols && row_dofs.data() == col_dofs.data();
    const bool has_constrained = constraints_ &&
        (hasConstrainedDof(row_dofs) || (!same_dofs && hasConstrainedDof(col_dofs)));

    if (!has_constrained) {
        if (matrix && !cell_matrix.empty()) {
            matrix->addValues(row_dofs, col_dofs, cell_matrix);
        }
        if (rhs && !cell_rhs.empty()) {
            rhs->addValues(row_dofs, cell_rhs);
        }
        return;
    }

    // A constrained line contributes through each of its masters. A free
    // DOF contributes through itself. A Dirichlet line has no masters.
    const auto for_masters = [](GlobalIndex dof, const auto& constraint, const auto& add) {
        if (constraint) {
            for (const auto& e : constraint->entries) add(e.master_dof, e.weight);
        } else {
            add(dof, 1.0);
        }
    };
    const auto keep = [this](double contribution) {
        return contribution != 0.0 && !(std::abs(contribution) < options_.zero_tolerance);
    };

    if (matrix && !cell_matrix.empty()) {
        for (std::size_t i = 0; i < n_rows; ++i) {
            const auto row = row_dofs[i];
            const auto row_constraint = constraints_->getConstraint(row);
            if (row_constraint && row_constraint->isDirichlet()) continue;
            for (std::size_t j = 0; j < n_cols; ++j) {
                const double value = cell_matrix[i * n_cols + j];
                if (value == 0.0) continue;
                const auto col = col_dofs[j];
                const auto col_constraint = constraints_->getConstraint(col);
                for_masters(row, row_constraint, [&](GlobalIndex r, double wr) {
                    for_masters(col, col_constraint, [&](GlobalIndex c, double wc) {
                        const double entry = wr * wc * value;
                        if (keep(entry)) matrix->addValue(r, c, entry);
                    });
                    if (rhs && options_.apply_inhomogeneities && col_constraint && !col_constraint->isHomogeneous()) {
                        const double lift = -wr * value * col_constraint->inhomogeneity;
                        if (keep(lift)) rhs->addValue(r, lift);
                    }
                });
            }
        }
    }
    if (rhs && !cell_rhs.empty()) {
        for (std::size_t i = 0; i < n_rows; ++i) {
            const auto row = row_dofs[i];
            if (cell_rhs[i] == 0.0) continue;
            for_masters(row, constraints_->getConstraint(row), [&](GlobalIndex r, double wr) {
                const double entry = wr * cell_rhs[i];
                if (keep(entry)) rhs->addValue(r, entry);
            });
        }
    }

    if (!finalize_rows) return;

    // Ensure constrained rows are well-posed (identity-like with RHS
    // inhomogeneity). This covers plain Dirichlet lines AND master-bearing
    // slave lines (hanging-node/MPC/aggregation): condensation distributes
    // their physical contributions to the masters and leaves the slave row
    // empty, which is singular for direct factorizations. The solve then
    // returns the inhomogeneity for the slave; the true slave value is
    // reconstructed by the post-solve constraint distribution.
    for (std::size_t i = 0; i < n_rows; ++i) {
        const auto dof = row_dofs[i];
        const auto constraint = constraints_->getConstraint(dof);
        if (!constraint) {
            continue;
        }

        if (matrix) {
            matrix->setDiagonal(dof, options_.constrained_diagonal);
        }
        if (rhs) {
            const double inhom = options_.apply_inhomogeneities ? constraint->inhomogeneity : 0.0;
            rhs->setValue(dof, inhom * options_.constrained_diagonal);
        }
    }
}

void ConstraintDistributor::distributeElementCore(
    std::span<const double> cell_matrix,
    std::span<const double> cell_rhs,
    std::span<const GlobalIndex> cell_dofs,
    IMatrixOperations* matrix,
    IVectorOperations* rhs) const
{
    distributeElementCoreRectangular(cell_matrix, cell_rhs, cell_dofs, cell_dofs, matrix, rhs);
}

bool ConstraintDistributor::hasConstrainedDof(std::span<const GlobalIndex> dofs) const {
    if (!constraints_) return false;
    return constraints_->hasConstrainedDofs(dofs);
}

// ============================================================================
// Batch distribution
// ============================================================================

void ConstraintDistributor::distributeLocalToGlobalBatch(
    GlobalIndex n_cells,
    std::span<const double> cell_matrices,
    std::span<const double> cell_rhs_vectors,
    std::span<const GlobalIndex> cell_dof_offsets,
    std::span<const GlobalIndex> cell_dof_ids,
    IMatrixOperations& matrix,
    IVectorOperations& rhs) const
{
    std::size_t matrix_offset = 0;
    std::size_t rhs_offset = 0;

    for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
        auto dof_start = static_cast<std::size_t>(cell_dof_offsets[static_cast<std::size_t>(cell)]);
        auto dof_end = static_cast<std::size_t>(cell_dof_offsets[static_cast<std::size_t>(cell + 1)]);
        std::size_t n_dofs = dof_end - dof_start;
        std::size_t matrix_size = n_dofs * n_dofs;

        std::span<const GlobalIndex> dofs(cell_dof_ids.data() + dof_start, n_dofs);
        std::span<const double> cell_mat(cell_matrices.data() + matrix_offset, matrix_size);
        std::span<const double> cell_rhs(cell_rhs_vectors.data() + rhs_offset, n_dofs);

        distributeLocalToGlobal(cell_mat, cell_rhs, dofs, matrix, rhs);

        matrix_offset += matrix_size;
        rhs_offset += n_dofs;
    }
}

void ConstraintDistributor::distributeLocalToGlobalUniform(
    GlobalIndex n_cells,
    LocalIndex dofs_per_cell,
    std::span<const double> cell_matrices,
    std::span<const double> cell_rhs_vectors,
    std::span<const GlobalIndex> cell_dof_ids,
    IMatrixOperations& matrix,
    IVectorOperations& rhs) const
{
    std::size_t matrix_size = static_cast<std::size_t>(dofs_per_cell) * dofs_per_cell;

    for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
        std::size_t dof_offset = static_cast<std::size_t>(cell) * dofs_per_cell;
        std::size_t mat_offset = static_cast<std::size_t>(cell) * matrix_size;
        std::size_t rhs_offset = dof_offset;

        std::span<const GlobalIndex> dofs(cell_dof_ids.data() + dof_offset, dofs_per_cell);
        std::span<const double> cell_mat(cell_matrices.data() + mat_offset, matrix_size);
        std::span<const double> cell_rhs(cell_rhs_vectors.data() + rhs_offset, dofs_per_cell);

        distributeLocalToGlobal(cell_mat, cell_rhs, dofs, matrix, rhs);
    }
}

// ============================================================================
// Direct vector operations
// ============================================================================

void ConstraintDistributor::distributeSolution(IVectorOperations& solution) const {
    if (!constraints_) return;

    // For each constrained DOF, compute value from masters
    constraints_->forEach([&](const AffineConstraints::ConstraintView& view) {
        double value = view.inhomogeneity;
        for (const auto& entry : view.entries) {
            if (entry.master_dof < solution.size()) {
                value += entry.weight * solution.getValue(entry.master_dof);
            }
        }
        if (view.slave_dof < solution.size()) {
            solution.setValue(view.slave_dof, value);
        }
    });
}

void ConstraintDistributor::setConstrainedEntries(IVectorOperations& vec) const {
    if (!constraints_) return;

    constraints_->forEach([&](const AffineConstraints::ConstraintView& view) {
        if (view.slave_dof < vec.size()) {
            vec.setValue(view.slave_dof, view.inhomogeneity);
        }
    });
}

void ConstraintDistributor::zeroConstrainedEntries(IVectorOperations& vec) const {
    if (!constraints_) return;

    constraints_->forEach([&](const AffineConstraints::ConstraintView& view) {
        if (view.slave_dof < vec.size()) {
            vec.setValue(view.slave_dof, 0.0);
        }
    });
}

// ============================================================================
// Condensation
// ============================================================================

void ConstraintDistributor::condenseLocal(
    std::vector<double>& cell_matrix,
    std::vector<double>& cell_rhs,
    std::span<const GlobalIndex> cell_dofs) const
{
    if (!constraints_) return;

    const std::size_t n_dofs = cell_dofs.size();

    // For each constrained DOF in the element, condense its contributions
    for (std::size_t i = 0; i < n_dofs; ++i) {
        auto constraint = constraints_->getConstraint(cell_dofs[i]);
        if (!constraint) continue;

        if (constraint->isDirichlet()) {
            // Dirichlet: eliminate from other rows, zero row and column, set diagonal to 1
            double inhom = constraint->inhomogeneity;
            
            // Eliminate from other rows' RHS
            for (std::size_t j = 0; j < n_dofs; ++j) {
                if (i == j) continue;
                cell_rhs[j] -= cell_matrix[j * n_dofs + i] * inhom;
            }

            // Zero row and column
            for (std::size_t j = 0; j < n_dofs; ++j) {
                cell_matrix[i * n_dofs + j] = 0.0;
                cell_matrix[j * n_dofs + i] = 0.0;
            }
            cell_matrix[i * n_dofs + i] = options_.constrained_diagonal;

            // RHS gets inhomogeneity (scaled by diagonal)
            cell_rhs[i] = inhom * options_.constrained_diagonal;
        } else {
            // MPC: redistribute to masters (complex - simplified here)
            // Full implementation would modify matrix entries
            // For now, just zero the constrained row/col
            for (std::size_t j = 0; j < n_dofs; ++j) {
                cell_matrix[i * n_dofs + j] = 0.0;
                cell_matrix[j * n_dofs + i] = 0.0;
            }
            cell_matrix[i * n_dofs + i] = options_.constrained_diagonal;
            cell_rhs[i] = 0.0;
        }
    }
}

// ============================================================================
// DenseMatrixOps implementation
// ============================================================================

DenseMatrixOps::DenseMatrixOps(GlobalIndex n_rows, GlobalIndex n_cols)
    : n_rows_(n_rows), n_cols_(n_cols),
      data_(static_cast<std::size_t>(n_rows * n_cols), 0.0) {}

void DenseMatrixOps::addValues(std::span<const GlobalIndex> rows,
                                std::span<const GlobalIndex> cols,
                                std::span<const double> values) {
    const std::size_t n_rows = rows.size();
    const std::size_t n_cols = cols.size();

    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < n_cols; ++j) {
            if (rows[i] >= 0 && rows[i] < n_rows_ &&
                cols[j] >= 0 && cols[j] < n_cols_) {
                data_[static_cast<std::size_t>(rows[i] * n_cols_ + cols[j])] +=
                    values[i * n_cols + j];
            }
        }
    }
}

void DenseMatrixOps::addValue(GlobalIndex row, GlobalIndex col, double value) {
    if (row >= 0 && row < n_rows_ && col >= 0 && col < n_cols_) {
        data_[static_cast<std::size_t>(row * n_cols_ + col)] += value;
    }
}

void DenseMatrixOps::setDiagonal(GlobalIndex row, double value) {
    if (row >= 0 && row < n_rows_ && row < n_cols_) {
        data_[static_cast<std::size_t>(row * n_cols_ + row)] = value;
    }
}

void DenseMatrixOps::clear() {
    std::fill(data_.begin(), data_.end(), 0.0);
}

// ============================================================================
// DenseVectorOps implementation
// ============================================================================

DenseVectorOps::DenseVectorOps(GlobalIndex size)
    : data_(static_cast<std::size_t>(size), 0.0) {}

void DenseVectorOps::addValues(std::span<const GlobalIndex> indices,
                                std::span<const double> values) {
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= 0 && static_cast<std::size_t>(indices[i]) < data_.size()) {
            data_[static_cast<std::size_t>(indices[i])] += values[i];
        }
    }
}

void DenseVectorOps::addValue(GlobalIndex index, double value) {
    if (index >= 0 && static_cast<std::size_t>(index) < data_.size()) {
        data_[static_cast<std::size_t>(index)] += value;
    }
}

void DenseVectorOps::setValue(GlobalIndex index, double value) {
    if (index >= 0 && static_cast<std::size_t>(index) < data_.size()) {
        data_[static_cast<std::size_t>(index)] = value;
    }
}

double DenseVectorOps::getValue(GlobalIndex index) const {
    if (index >= 0 && static_cast<std::size_t>(index) < data_.size()) {
        return data_[static_cast<std::size_t>(index)];
    }
    return 0.0;
}

void DenseVectorOps::clear() {
    std::fill(data_.begin(), data_.end(), 0.0);
}

} // namespace constraints
} // namespace FE
} // namespace svmp
