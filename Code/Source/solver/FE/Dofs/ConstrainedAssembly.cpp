/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "ConstrainedAssembly.h"
#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace dofs {

// =============================================================================
// ConstrainedAssembly Implementation
// =============================================================================

ConstrainedAssembly::ConstrainedAssembly() = default;
ConstrainedAssembly::~ConstrainedAssembly() = default;

ConstrainedAssembly::ConstrainedAssembly(ConstrainedAssembly&&) noexcept = default;
ConstrainedAssembly& ConstrainedAssembly::operator=(ConstrainedAssembly&&) noexcept = default;

void ConstrainedAssembly::initialize(const DofConstraints& constraints,
                                      const ConstrainedAssemblyOptions& options) {
    constraints_ = &constraints;
    options_ = options;
}

void ConstrainedAssembly::setOptions(const ConstrainedAssemblyOptions& options) {
    options_ = options;
}

void ConstrainedAssembly::distributeLocalToGlobal(
    std::span<const double> cell_matrix,
    std::span<const double> cell_rhs,
    std::span<const GlobalIndex> cell_dof_ids,
    BackendAdapter& adapter) const {

    if (!constraints_) {
        // No constraints - direct assembly
        adapter.addMatrixValues(cell_dof_ids, cell_dof_ids, cell_matrix);
        adapter.addVectorValues(cell_dof_ids, cell_rhs);
        return;
    }

    // Check if any DOFs are constrained
    bool has_constrained = false;
    for (auto dof : cell_dof_ids) {
        if (constraints_->isConstrained(dof)) {
            has_constrained = true;
            break;
        }
    }

    if (!has_constrained) {
        // No constrained DOFs - direct assembly
        adapter.addMatrixValues(cell_dof_ids, cell_dof_ids, cell_matrix);
        adapter.addVectorValues(cell_dof_ids, cell_rhs);
        return;
    }

    // Apply element-level constraint modification
    work_rows_.clear();
    work_cols_.clear();
    work_values_.clear();
    work_rhs_.clear();

    std::vector<GlobalIndex> rhs_indices;

    applyElementConstraints(cell_matrix, cell_rhs, cell_dof_ids,
                            work_rows_, work_cols_, work_values_,
                            rhs_indices, work_rhs_);

    // Assemble modified contributions
    if (!work_values_.empty()) {
        adapter.addMatrixValuesSparse(work_rows_, work_cols_, work_values_);
    }
    if (!work_rhs_.empty()) {
        adapter.addVectorValues(rhs_indices, work_rhs_);
    }
}

void ConstrainedAssembly::distributeMatrixToGlobal(
    std::span<const double> cell_matrix,
    std::span<const GlobalIndex> cell_dof_ids,
    BackendAdapter& adapter) const {

    // Create zero RHS
    std::vector<double> zero_rhs(cell_dof_ids.size(), 0.0);
    distributeLocalToGlobal(cell_matrix, zero_rhs, cell_dof_ids, adapter);
}

void ConstrainedAssembly::distributeRhsToGlobal(
    std::span<const double> cell_rhs,
    std::span<const GlobalIndex> cell_dof_ids,
    BackendAdapter& adapter) const {

    if (!constraints_) {
        adapter.addVectorValues(cell_dof_ids, cell_rhs);
        return;
    }

    // Apply constraints to RHS
    std::vector<GlobalIndex> out_indices;
    std::vector<double> out_values;

    for (std::size_t i = 0; i < cell_dof_ids.size(); ++i) {
        GlobalIndex dof = cell_dof_ids[i];

        if (constraints_->isConstrained(dof)) {
            // Get constraint line
            auto constraint_opt = constraints_->getConstraintLine(dof);
            if (constraint_opt && !constraint_opt->isDirichlet()) {
                // Distribute to master DOFs
                double val = cell_rhs[i];
                for (const auto& entry : constraint_opt->entries) {
                    out_indices.push_back(entry.dof);
                    out_values.push_back(entry.coefficient * val);
                }
            }
            // For Dirichlet, RHS contribution is dropped
        } else {
            out_indices.push_back(dof);
            out_values.push_back(cell_rhs[i]);
        }
    }

    if (!out_values.empty()) {
        adapter.addVectorValues(out_indices, out_values);
    }
}

void ConstrainedAssembly::distributeLocalToGlobalBatch(
    GlobalIndex n_cells,
    std::span<const double> cell_matrices,
    std::span<const double> cell_rhs_vectors,
    std::span<const GlobalIndex> cell_dof_offsets,
    std::span<const GlobalIndex> cell_dof_ids,
    BackendAdapter& adapter) const {

    adapter.beginAssembly();

    GlobalIndex matrix_offset = 0;
    GlobalIndex rhs_offset = 0;

    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto dof_start = static_cast<std::size_t>(cell_dof_offsets[static_cast<std::size_t>(c)]);
        auto dof_end = static_cast<std::size_t>(cell_dof_offsets[static_cast<std::size_t>(c) + 1]);
        auto n_dofs = static_cast<LocalIndex>(dof_end - dof_start);

        std::span<const GlobalIndex> dofs(cell_dof_ids.data() + dof_start, n_dofs);
        std::span<const double> matrix(cell_matrices.data() + matrix_offset, n_dofs * n_dofs);
        std::span<const double> rhs(cell_rhs_vectors.data() + rhs_offset, n_dofs);

        distributeLocalToGlobal(matrix, rhs, dofs, adapter);

        matrix_offset += n_dofs * n_dofs;
        rhs_offset += n_dofs;
    }

    adapter.endAssembly();
}

void ConstrainedAssembly::distributeLocalToGlobalUniform(
    GlobalIndex n_cells,
    LocalIndex dofs_per_cell,
    std::span<const double> cell_matrices,
    std::span<const double> cell_rhs_vectors,
    std::span<const GlobalIndex> cell_dof_ids,
    BackendAdapter& adapter) const {

    adapter.beginAssembly();

    auto mat_size = static_cast<std::size_t>(dofs_per_cell) * dofs_per_cell;
    auto rhs_size = static_cast<std::size_t>(dofs_per_cell);

    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto idx = static_cast<std::size_t>(c);

        std::span<const GlobalIndex> dofs(cell_dof_ids.data() + idx * dofs_per_cell, dofs_per_cell);
        std::span<const double> matrix(cell_matrices.data() + idx * mat_size, mat_size);
        std::span<const double> rhs(cell_rhs_vectors.data() + idx * rhs_size, rhs_size);

        distributeLocalToGlobal(matrix, rhs, dofs, adapter);
    }

    adapter.endAssembly();
}

void ConstrainedAssembly::applyElementConstraints(
    std::span<const double> cell_matrix,
    std::span<const double> cell_rhs,
    std::span<const GlobalIndex> cell_dof_ids,
    std::vector<GlobalIndex>& out_rows,
    std::vector<GlobalIndex>& out_cols,
    std::vector<double>& out_values,
    std::vector<GlobalIndex>& out_rhs_indices,
    std::vector<double>& out_rhs_values) const {

    auto n = static_cast<LocalIndex>(cell_dof_ids.size());

    // Process each entry in the element matrix
    for (LocalIndex i = 0; i < n; ++i) {
        GlobalIndex row_dof = cell_dof_ids[static_cast<std::size_t>(i)];
        double rhs_val = cell_rhs[static_cast<std::size_t>(i)];

        auto row_constraint = constraints_->getConstraintLine(row_dof);

        for (LocalIndex j = 0; j < n; ++j) {
            GlobalIndex col_dof = cell_dof_ids[static_cast<std::size_t>(j)];
            double mat_val = cell_matrix[static_cast<std::size_t>(i * n + j)];

            if (std::abs(mat_val) < 1e-15) continue;

            auto col_constraint = constraints_->getConstraintLine(col_dof);

            if (!row_constraint && !col_constraint) {
                // No constraints - direct assembly
                out_rows.push_back(row_dof);
                out_cols.push_back(col_dof);
                out_values.push_back(mat_val);
            } else if (row_constraint && !col_constraint) {
                // Row constrained: distribute to master DOFs
                if (!row_constraint->isDirichlet()) {
                    for (const auto& entry : row_constraint->entries) {
                        out_rows.push_back(entry.dof);
                        out_cols.push_back(col_dof);
                        out_values.push_back(entry.coefficient * mat_val);
                    }
                }
            } else if (!row_constraint && col_constraint) {
                // Column constrained
                if (!col_constraint->isDirichlet()) {
                    for (const auto& entry : col_constraint->entries) {
                        out_rows.push_back(row_dof);
                        out_cols.push_back(entry.dof);
                        out_values.push_back(entry.coefficient * mat_val);
                    }
                }
                // Add inhomogeneity contribution to RHS
                if (std::abs(col_constraint->inhomogeneity) > 1e-15) {
                    rhs_val -= mat_val * col_constraint->inhomogeneity;
                }
            } else {
                // Both constrained
                if (!row_constraint->isDirichlet() && !col_constraint->isDirichlet()) {
                    for (const auto& row_entry : row_constraint->entries) {
                        for (const auto& col_entry : col_constraint->entries) {
                            out_rows.push_back(row_entry.dof);
                            out_cols.push_back(col_entry.dof);
                            out_values.push_back(
                                row_entry.coefficient * col_entry.coefficient * mat_val);
                        }
                    }
                }
            }
        }

        // Handle RHS
        if (!row_constraint) {
            out_rhs_indices.push_back(row_dof);
            out_rhs_values.push_back(rhs_val);
        } else if (!row_constraint->isDirichlet()) {
            for (const auto& entry : row_constraint->entries) {
                out_rhs_indices.push_back(entry.dof);
                out_rhs_values.push_back(entry.coefficient * rhs_val);
            }
        }
    }
}

// =============================================================================
// DenseMatrixAdapter Implementation
// =============================================================================

DenseMatrixAdapter::DenseMatrixAdapter(GlobalIndex n_rows, GlobalIndex n_cols)
    : n_rows_(n_rows)
    , n_cols_(n_cols)
    , matrix_(static_cast<std::size_t>(n_rows * n_cols), 0.0)
    , vector_(static_cast<std::size_t>(n_rows), 0.0) {}

void DenseMatrixAdapter::addMatrixValues(std::span<const GlobalIndex> rows,
                                          std::span<const GlobalIndex> cols,
                                          std::span<const double> values) {
    auto n_rows = static_cast<LocalIndex>(rows.size());
    auto n_cols = static_cast<LocalIndex>(cols.size());

    for (LocalIndex i = 0; i < n_rows; ++i) {
        for (LocalIndex j = 0; j < n_cols; ++j) {
            auto row = rows[static_cast<std::size_t>(i)];
            auto col = cols[static_cast<std::size_t>(j)];
            if (row >= 0 && row < n_rows_ && col >= 0 && col < n_cols_) {
                auto idx = static_cast<std::size_t>(row * n_cols_ + col);
                auto val_idx = static_cast<std::size_t>(i * n_cols + j);
                matrix_[idx] += values[val_idx];
            }
        }
    }
}

void DenseMatrixAdapter::addMatrixValuesSparse(std::span<const GlobalIndex> row_indices,
                                                std::span<const GlobalIndex> col_indices,
                                                std::span<const double> values) {
    for (std::size_t k = 0; k < values.size(); ++k) {
        auto row = row_indices[k];
        auto col = col_indices[k];
        if (row >= 0 && row < n_rows_ && col >= 0 && col < n_cols_) {
            auto idx = static_cast<std::size_t>(row * n_cols_ + col);
            matrix_[idx] += values[k];
        }
    }
}

void DenseMatrixAdapter::addVectorValues(std::span<const GlobalIndex> indices,
                                          std::span<const double> values) {
    for (std::size_t i = 0; i < indices.size() && i < values.size(); ++i) {
        auto idx = indices[i];
        if (idx >= 0 && idx < n_rows_) {
            vector_[static_cast<std::size_t>(idx)] += values[i];
        }
    }
}

void DenseMatrixAdapter::setVectorValues(std::span<const GlobalIndex> indices,
                                          std::span<const double> values) {
    for (std::size_t i = 0; i < indices.size() && i < values.size(); ++i) {
        auto idx = indices[i];
        if (idx >= 0 && idx < n_rows_) {
            vector_[static_cast<std::size_t>(idx)] = values[i];
        }
    }
}

double DenseMatrixAdapter::getMatrixEntry(GlobalIndex row, GlobalIndex col) const {
    if (row < 0 || row >= n_rows_ || col < 0 || col >= n_cols_) {
        return 0.0;
    }
    return matrix_[static_cast<std::size_t>(row * n_cols_ + col)];
}

double DenseMatrixAdapter::getVectorEntry(GlobalIndex index) const {
    if (index < 0 || index >= n_rows_) {
        return 0.0;
    }
    return vector_[static_cast<std::size_t>(index)];
}

// =============================================================================
// CSRMatrixAdapter Implementation
// =============================================================================

CSRMatrixAdapter::CSRMatrixAdapter(std::span<const GlobalIndex> row_offsets,
                                    std::span<const GlobalIndex> col_indices)
    : row_offsets_(row_offsets.begin(), row_offsets.end())
    , col_indices_(col_indices.begin(), col_indices.end())
    , values_(col_indices.size(), 0.0) {

    if (!row_offsets_.empty()) {
        vector_.resize(static_cast<std::size_t>(row_offsets_.size() - 1), 0.0);
    }
}

void CSRMatrixAdapter::addMatrixValues(std::span<const GlobalIndex> rows,
                                        std::span<const GlobalIndex> cols,
                                        std::span<const double> values) {
    auto n_rows = static_cast<LocalIndex>(rows.size());
    auto n_cols = static_cast<LocalIndex>(cols.size());

    for (LocalIndex i = 0; i < n_rows; ++i) {
        for (LocalIndex j = 0; j < n_cols; ++j) {
            auto row = rows[static_cast<std::size_t>(i)];
            auto col = cols[static_cast<std::size_t>(j)];
            auto val_idx = static_cast<std::size_t>(i * n_cols + j);

            auto csr_idx = findColIndex(row, col);
            if (csr_idx >= 0) {
                values_[static_cast<std::size_t>(csr_idx)] += values[val_idx];
            }
        }
    }
}

void CSRMatrixAdapter::addMatrixValuesSparse(std::span<const GlobalIndex> row_indices,
                                              std::span<const GlobalIndex> col_indices,
                                              std::span<const double> values) {
    for (std::size_t k = 0; k < values.size(); ++k) {
        auto csr_idx = findColIndex(row_indices[k], col_indices[k]);
        if (csr_idx >= 0) {
            values_[static_cast<std::size_t>(csr_idx)] += values[k];
        }
    }
}

void CSRMatrixAdapter::addVectorValues(std::span<const GlobalIndex> indices,
                                        std::span<const double> values) {
    for (std::size_t i = 0; i < indices.size() && i < values.size(); ++i) {
        auto idx = indices[i];
        if (idx >= 0 && static_cast<std::size_t>(idx) < vector_.size()) {
            vector_[static_cast<std::size_t>(idx)] += values[i];
        }
    }
}

void CSRMatrixAdapter::setVectorValues(std::span<const GlobalIndex> indices,
                                        std::span<const double> values) {
    for (std::size_t i = 0; i < indices.size() && i < values.size(); ++i) {
        auto idx = indices[i];
        if (idx >= 0 && static_cast<std::size_t>(idx) < vector_.size()) {
            vector_[static_cast<std::size_t>(idx)] = values[i];
        }
    }
}

GlobalIndex CSRMatrixAdapter::findColIndex(GlobalIndex row, GlobalIndex col) const {
    if (row < 0 || static_cast<std::size_t>(row + 1) >= row_offsets_.size()) {
        return -1;
    }

    auto start = static_cast<std::size_t>(row_offsets_[static_cast<std::size_t>(row)]);
    auto end = static_cast<std::size_t>(row_offsets_[static_cast<std::size_t>(row) + 1]);

    for (std::size_t i = start; i < end && i < col_indices_.size(); ++i) {
        if (col_indices_[i] == col) {
            return static_cast<GlobalIndex>(i);
        }
    }

    return -1;  // Column not in sparsity pattern
}

} // namespace dofs
} // namespace FE
} // namespace svmp
