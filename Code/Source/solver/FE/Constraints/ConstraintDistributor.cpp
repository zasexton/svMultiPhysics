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

void ConstraintDistributor::distributeElementCoreRectangular(
    std::span<const double> cell_matrix,
    std::span<const double> cell_rhs,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    IMatrixOperations* matrix,
    IVectorOperations* rhs) const
{
    const std::size_t n_rows = row_dofs.size();
    const std::size_t n_cols = col_dofs.size();

    const bool has_constrained =
        hasConstrainedDof(row_dofs) || hasConstrainedDof(col_dofs);

    if (!has_constrained) {
        if (matrix && !cell_matrix.empty()) {
            matrix->addValues(row_dofs, col_dofs, cell_matrix);
        }
        if (rhs && !cell_rhs.empty()) {
            rhs->addValues(row_dofs, cell_rhs);
        }
        return;
    }

    if (matrix && !cell_matrix.empty()) {
        for (std::size_t i = 0; i < n_rows; ++i) {
            const GlobalIndex row_dof = row_dofs[i];
            const auto row_constraint = constraints_->getConstraint(row_dof);

            for (std::size_t j = 0; j < n_cols; ++j) {
                const GlobalIndex col_dof = col_dofs[j];
                const double value = cell_matrix[i * n_cols + j];

                if (std::abs(value) < options_.zero_tolerance) {
                    continue;
                }

                const auto col_constraint = constraints_->getConstraint(col_dof);

                if (!row_constraint && !col_constraint) {
                    matrix->addValue(row_dof, col_dof, value);
                }
                else if (row_constraint && !col_constraint) {
                    if (row_constraint->isDirichlet()) {
                        continue;
                    }
                    for (const auto& entry : row_constraint->entries) {
                        matrix->addValue(entry.master_dof, col_dof,
                                         entry.weight * value);
                    }
                }
                else if (!row_constraint && col_constraint) {
                    if (col_constraint->isDirichlet()) {
                        if (rhs && options_.apply_inhomogeneities) {
                            const double inhom = col_constraint->inhomogeneity;
                            if (std::abs(inhom) > options_.zero_tolerance) {
                                rhs->addValue(row_dof, -value * inhom);
                            }
                        }
                        continue;
                    }
                    if (options_.symmetric) {
                        for (const auto& entry : col_constraint->entries) {
                            matrix->addValue(row_dof, entry.master_dof,
                                             entry.weight * value);
                        }
                        if (rhs && options_.apply_inhomogeneities) {
                            const double inhom = col_constraint->inhomogeneity;
                            if (std::abs(inhom) > options_.zero_tolerance) {
                                rhs->addValue(row_dof, -value * inhom);
                            }
                        }
                    }
                }
                else {
                    if (row_constraint->isDirichlet()) {
                        continue;
                    }
                    if (col_constraint->isDirichlet()) {
                        if (rhs && options_.apply_inhomogeneities) {
                            const double inhom = col_constraint->inhomogeneity;
                            if (std::abs(inhom) > options_.zero_tolerance) {
                                for (const auto& r_entry : row_constraint->entries) {
                                    rhs->addValue(r_entry.master_dof,
                                                  -r_entry.weight * value * inhom);
                                }
                            }
                        }
                        continue;
                    }

                    for (const auto& r_entry : row_constraint->entries) {
                        for (const auto& c_entry : col_constraint->entries) {
                            const double combined_weight =
                                r_entry.weight * c_entry.weight * value;
                            if (std::abs(combined_weight) >= options_.zero_tolerance) {
                                matrix->addValue(r_entry.master_dof, c_entry.master_dof,
                                                 combined_weight);
                            }
                        }
                        if (rhs && options_.apply_inhomogeneities) {
                            const double inhom = col_constraint->inhomogeneity;
                            if (std::abs(inhom) > options_.zero_tolerance) {
                                rhs->addValue(r_entry.master_dof,
                                              -r_entry.weight * value * inhom);
                            }
                        }
                    }
                }
            }
        }
    }

    if (rhs && !cell_rhs.empty()) {
        for (std::size_t i = 0; i < n_rows; ++i) {
            const GlobalIndex dof = row_dofs[i];
            const double value = cell_rhs[i];

            if (std::abs(value) < options_.zero_tolerance) {
                continue;
            }

            const auto constraint = constraints_->getConstraint(dof);

            if (!constraint) {
                rhs->addValue(dof, value);
            } else if (constraint->isDirichlet()) {
                continue;
            } else {
                for (const auto& entry : constraint->entries) {
                    rhs->addValue(entry.master_dof, entry.weight * value);
                }
            }
        }
    }

    // Ensure Dirichlet rows are well-posed (identity-like with RHS inhomogeneity).
    for (std::size_t i = 0; i < n_rows; ++i) {
        const auto dof = row_dofs[i];
        const auto constraint = constraints_->getConstraint(dof);
        if (!constraint || !constraint->isDirichlet()) {
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
    const std::size_t n_dofs = cell_dofs.size();

    // Check if there are any constrained DOFs in this element
    bool has_constrained = hasConstrainedDof(cell_dofs);

    if (!has_constrained) {
        // Fast path: no constraints, direct distribution
        if (matrix && !cell_matrix.empty()) {
            matrix->addValues(cell_dofs, cell_dofs, cell_matrix);
        }
        if (rhs && !cell_rhs.empty()) {
            rhs->addValues(cell_dofs, cell_rhs);
        }
        return;
    }

    // Slow path: handle constraints
    // For each pair (i, j) in the element matrix:
    //   - If neither i nor j is constrained: add directly
    //   - If i is constrained: distribute to i's masters
    //   - If j is constrained: distribute to j's masters (symmetric case)
    //   - If both: distribute to cross-product of masters

    // Process matrix entries
    if (matrix && !cell_matrix.empty()) {
        for (std::size_t i = 0; i < n_dofs; ++i) {
            GlobalIndex row_dof = cell_dofs[i];
            auto row_constraint = constraints_->getConstraint(row_dof);

            for (std::size_t j = 0; j < n_dofs; ++j) {
                GlobalIndex col_dof = cell_dofs[j];
                double value = cell_matrix[i * n_dofs + j];

                if (std::abs(value) < options_.zero_tolerance) {
                    continue;
                }

                auto col_constraint = constraints_->getConstraint(col_dof);

                if (!row_constraint && !col_constraint) {
                    // Neither constrained - direct add
                    matrix->addValue(row_dof, col_dof, value);
                }
                else if (row_constraint && !col_constraint) {
                    // Row constrained: distribute to masters
                    if (row_constraint->isDirichlet()) {
                        // Dirichlet: skip (row will be set to identity)
                        continue;
                    }
                    for (const auto& entry : row_constraint->entries) {
                        matrix->addValue(entry.master_dof, col_dof,
                                         entry.weight * value);
                    }
                }
                else if (!row_constraint && col_constraint) {
                    // Column constrained: distribute to masters (for symmetric)
                    if (col_constraint->isDirichlet()) {
                        // Inhomogeneity contribution to RHS
                        if (rhs && options_.apply_inhomogeneities) {
                            double inhom = col_constraint->inhomogeneity;
                            if (std::abs(inhom) > options_.zero_tolerance) {
                                rhs->addValue(row_dof, -value * inhom);
                            }
                        }
                        continue;
                    }
                    if (options_.symmetric) {
                        for (const auto& entry : col_constraint->entries) {
                            matrix->addValue(row_dof, entry.master_dof,
                                             entry.weight * value);
                        }
                        // Inhomogeneity contribution
                        if (rhs && options_.apply_inhomogeneities) {
                            double inhom = col_constraint->inhomogeneity;
                            if (std::abs(inhom) > options_.zero_tolerance) {
                                rhs->addValue(row_dof, -value * inhom);
                            }
                        }
                    }
                }
                else {
                    // Both constrained
                    if (row_constraint->isDirichlet()) {
                        continue;  // Row will be set to identity
                    }
                    if (col_constraint->isDirichlet()) {
                        // Row masters get inhomogeneity contribution
                        if (rhs && options_.apply_inhomogeneities) {
                            double inhom = col_constraint->inhomogeneity;
                            if (std::abs(inhom) > options_.zero_tolerance) {
                                for (const auto& r_entry : row_constraint->entries) {
                                    rhs->addValue(r_entry.master_dof,
                                                  -r_entry.weight * value * inhom);
                                }
                            }
                        }
                        continue;
                    }

                    // Both have masters: cross-product
                    for (const auto& r_entry : row_constraint->entries) {
                        for (const auto& c_entry : col_constraint->entries) {
                            double combined_weight = r_entry.weight * c_entry.weight * value;
                            if (std::abs(combined_weight) >= options_.zero_tolerance) {
                                matrix->addValue(r_entry.master_dof, c_entry.master_dof,
                                                 combined_weight);
                            }
                        }
                        // Inhomogeneity contribution from column
                        if (rhs && options_.apply_inhomogeneities) {
                            double inhom = col_constraint->inhomogeneity;
                            if (std::abs(inhom) > options_.zero_tolerance) {
                                rhs->addValue(r_entry.master_dof,
                                              -r_entry.weight * value * inhom);
                            }
                        }
                    }
                }
            }
        }
    }

    // Process RHS entries
    if (rhs && !cell_rhs.empty()) {
        for (std::size_t i = 0; i < n_dofs; ++i) {
            GlobalIndex dof = cell_dofs[i];
            double value = cell_rhs[i];

            if (std::abs(value) < options_.zero_tolerance) {
                continue;
            }

            auto constraint = constraints_->getConstraint(dof);

            if (!constraint) {
                // Not constrained - direct add
                rhs->addValue(dof, value);
            } else if (constraint->isDirichlet()) {
                // Dirichlet: skip (will be set to inhomogeneity)
                continue;
            } else {
                // Distribute to masters
                for (const auto& entry : constraint->entries) {
                    rhs->addValue(entry.master_dof, entry.weight * value);
                }
            }
        }
    }

    // Ensure constrained rows are well-posed.
    //
    // Dirichlet rows/cols were skipped above (to eliminate couplings) under the
    // assumption that the constrained rows are set to identity-like form.
    // Without this post-step, constrained rows can remain entirely zero.
    //
    // For matrix-only assembly, setDiagonal is sufficient; for RHS assembly, set
    // constrained entries to the inhomogeneity value (scaled by diagonal).
    for (std::size_t i = 0; i < n_dofs; ++i) {
        const auto dof = cell_dofs[i];
        auto constraint = constraints_->getConstraint(dof);
        if (!constraint || !constraint->isDirichlet()) {
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
        auto dof_start = static_cast<std::size_t>(cell_dof_offsets[cell]);
        auto dof_end = static_cast<std::size_t>(cell_dof_offsets[cell + 1]);
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
