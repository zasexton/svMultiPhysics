/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "ConstrainedAssembly.h"
#include <algorithm>
#include <optional>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace dofs {
namespace {

std::optional<ConstraintLine> constraintLineForAssembly(const DofConstraints& constraints, GlobalIndex dof) {
    auto line = constraints.getConstraintLine(dof);
    if (line && !constraints.isClosed()) {
        for (const auto& master : line->entries) {
            if (constraints.isConstrained(master.dof)) {
                throw FEException("ConstrainedAssembly: DOF " + std::to_string(dof) +
                    " depends on constrained master " + std::to_string(master.dof) +
                    ". Call DofConstraints::close() before assembly.");
            }
        }
    }
    return line;
}

void checkOpenBatchConstraints(const DofConstraints* constraints, std::span<const GlobalIndex> dofs) {
    if (constraints && !constraints->isClosed()) {
        for (const auto dof : dofs) {
            constraintLineForAssembly(*constraints, dof);
        }
    }
}

} // namespace

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

    const bool constrained = constraints_ && std::any_of(cell_dof_ids.begin(), cell_dof_ids.end(),
        [&](GlobalIndex dof) { return constraints_->isConstrained(dof); });
    if (!constrained) {
        if (!cell_matrix.empty()) adapter.addMatrixValues(cell_dof_ids, cell_dof_ids, cell_matrix);
        if (!cell_rhs.empty()) adapter.addVectorValues(cell_dof_ids, cell_rhs);
        return;
    }

    // Each line represents x_i = P_i y + g_i. A free DOF has an identity
    // row in P, and a fixed DOF has an empty row and its prescribed offset.
    const auto n = cell_dof_ids.size();
    std::vector<ConstraintLine> lines;
    lines.reserve(n);
    for (const auto dof : cell_dof_ids) {
        if (auto constraint = constraintLineForAssembly(*constraints_, dof)) {
            lines.push_back(std::move(*constraint));
        } else {
            lines.push_back({dof, {{dof, 1.0}}});
        }
    }

    work_rows_.clear();
    work_cols_.clear();
    work_values_.clear();
    work_rhs_indices_.clear();
    work_rhs_.clear();

    for (std::size_t i = 0; i < n; ++i) {
        const auto& test = lines[i].entries;
        if (test.empty()) continue;

        double load = cell_rhs.empty() ? 0.0 : cell_rhs[i];
        if (!cell_matrix.empty()) {
            for (std::size_t j = 0; j < n; ++j) {
                const double value = cell_matrix[i * n + j];
                if (value == 0.0) continue;
                const auto& trial = lines[j];
                if (!cell_rhs.empty()) load -= value * trial.inhomogeneity;

                for (const auto& row : test) {
                    for (const auto& col : trial.entries) {
                        work_rows_.push_back(row.dof);
                        work_cols_.push_back(col.dof);
                        work_values_.push_back(row.coefficient * value * col.coefficient);
                    }
                }
            }
        }
        if (!cell_rhs.empty()) {
            for (const auto& row : test) {
                work_rhs_indices_.push_back(row.dof);
                work_rhs_.push_back(row.coefficient * load);
            }
        }
    }

    if (!work_values_.empty()) adapter.addMatrixValuesSparse(work_rows_, work_cols_, work_values_);
    if (!work_rhs_.empty()) adapter.addVectorValues(work_rhs_indices_, work_rhs_);
}

void ConstrainedAssembly::distributeMatrixToGlobal(
    std::span<const double> cell_matrix,
    std::span<const GlobalIndex> cell_dof_ids,
    BackendAdapter& adapter) const {
    distributeLocalToGlobal(cell_matrix, {}, cell_dof_ids, adapter);
}

void ConstrainedAssembly::distributeRhsToGlobal(
    std::span<const double> cell_rhs,
    std::span<const GlobalIndex> cell_dof_ids,
    BackendAdapter& adapter) const {
    distributeLocalToGlobal({}, cell_rhs, cell_dof_ids, adapter);
}

void ConstrainedAssembly::distributeLocalToGlobalBatch(
    GlobalIndex n_cells,
    std::span<const double> cell_matrices,
    std::span<const double> cell_rhs_vectors,
    std::span<const GlobalIndex> cell_dof_offsets,
    std::span<const GlobalIndex> cell_dof_ids,
    BackendAdapter& adapter) const {

    if (n_cells > 0) {
        const auto first = static_cast<std::size_t>(cell_dof_offsets.front());
        const auto end = static_cast<std::size_t>(cell_dof_offsets[static_cast<std::size_t>(n_cells)]);
        checkOpenBatchConstraints(constraints_, cell_dof_ids.subspan(first, end - first));
    }
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

    if (n_cells > 0) {
        checkOpenBatchConstraints(constraints_, cell_dof_ids.first(static_cast<std::size_t>(n_cells) * dofs_per_cell));
    }
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
