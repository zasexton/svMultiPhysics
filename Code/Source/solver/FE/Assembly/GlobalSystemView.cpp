/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "GlobalSystemView.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// DenseMatrixView Implementation
// ============================================================================

DenseMatrixView::DenseMatrixView(GlobalIndex n_dofs)
    : n_rows_(n_dofs)
    , n_cols_(n_dofs)
    , data_(static_cast<std::size_t>(n_dofs * n_dofs), 0.0)
{
    if (n_dofs < 0) {
        throw std::invalid_argument("DenseMatrixView: negative size");
    }
}

DenseMatrixView::DenseMatrixView(GlobalIndex n_rows, GlobalIndex n_cols)
    : n_rows_(n_rows)
    , n_cols_(n_cols)
    , data_(static_cast<std::size_t>(n_rows * n_cols), 0.0)
{
    if (n_rows < 0 || n_cols < 0) {
        throw std::invalid_argument("DenseMatrixView: negative dimensions");
    }
}

void DenseMatrixView::addMatrixEntries(
    std::span<const GlobalIndex> dofs,
    std::span<const Real> local_matrix,
    AddMode mode)
{
    addMatrixEntries(dofs, dofs, local_matrix, mode);
}

void DenseMatrixView::addMatrixEntries(
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    std::span<const Real> local_matrix,
    AddMode mode)
{
    const auto n_rows = static_cast<GlobalIndex>(row_dofs.size());
    const auto n_cols = static_cast<GlobalIndex>(col_dofs.size());

    if (local_matrix.size() != static_cast<std::size_t>(n_rows * n_cols)) {
        throw std::invalid_argument(
            "DenseMatrixView::addMatrixEntries: local_matrix size mismatch");
    }

    for (GlobalIndex i = 0; i < n_rows; ++i) {
        const GlobalIndex row = row_dofs[static_cast<std::size_t>(i)];
        if (row < 0 || row >= n_rows_) continue;

        for (GlobalIndex j = 0; j < n_cols; ++j) {
            const GlobalIndex col = col_dofs[static_cast<std::size_t>(j)];
            if (col < 0 || col >= n_cols_) continue;

            const auto local_idx = static_cast<std::size_t>(i * n_cols + j);
            const auto global_idx = static_cast<std::size_t>(row * n_cols_ + col);
            const Real value = local_matrix[local_idx];

            switch (mode) {
                case AddMode::Add:
                    data_[global_idx] += value;
                    break;
                case AddMode::Insert:
                    data_[global_idx] = value;
                    break;
                case AddMode::Max:
                    data_[global_idx] = std::max(data_[global_idx], value);
                    break;
                case AddMode::Min:
                    data_[global_idx] = std::min(data_[global_idx], value);
                    break;
            }
        }
    }
}

void DenseMatrixView::addMatrixEntry(
    GlobalIndex row,
    GlobalIndex col,
    Real value,
    AddMode mode)
{
    if (row < 0 || row >= n_rows_ || col < 0 || col >= n_cols_) {
        return;  // Silently ignore out-of-range entries
    }

    const auto idx = static_cast<std::size_t>(row * n_cols_ + col);

    switch (mode) {
        case AddMode::Add:
            data_[idx] += value;
            break;
        case AddMode::Insert:
            data_[idx] = value;
            break;
        case AddMode::Max:
            data_[idx] = std::max(data_[idx], value);
            break;
        case AddMode::Min:
            data_[idx] = std::min(data_[idx], value);
            break;
    }
}

void DenseMatrixView::setDiagonal(
    std::span<const GlobalIndex> dofs,
    std::span<const Real> values)
{
    if (dofs.size() != values.size()) {
        throw std::invalid_argument("DenseMatrixView::setDiagonal: size mismatch");
    }

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        setDiagonal(dofs[i], values[i]);
    }
}

void DenseMatrixView::setDiagonal(GlobalIndex dof, Real value)
{
    if (dof < 0 || dof >= std::min(n_rows_, n_cols_)) {
        return;
    }
    const auto idx = static_cast<std::size_t>(dof * n_cols_ + dof);
    data_[idx] = value;
}

void DenseMatrixView::zeroRows(
    std::span<const GlobalIndex> rows,
    bool set_diagonal)
{
    for (const GlobalIndex row : rows) {
        if (row < 0 || row >= n_rows_) continue;

        for (GlobalIndex col = 0; col < n_cols_; ++col) {
            const auto idx = static_cast<std::size_t>(row * n_cols_ + col);
            data_[idx] = 0.0;
        }

        if (set_diagonal && row < n_cols_) {
            const auto diag_idx = static_cast<std::size_t>(row * n_cols_ + row);
            data_[diag_idx] = 1.0;
        }
    }
}

// Vector operations (no-op for matrix-only)
void DenseMatrixView::addVectorEntries(
    std::span<const GlobalIndex> /*dofs*/,
    std::span<const Real> /*local_vector*/,
    AddMode /*mode*/)
{
    // No-op: matrix-only view
}

void DenseMatrixView::addVectorEntry(
    GlobalIndex /*dof*/,
    Real /*value*/,
    AddMode /*mode*/)
{
    // No-op
}

void DenseMatrixView::setVectorEntries(
    std::span<const GlobalIndex> /*dofs*/,
    std::span<const Real> /*values*/)
{
    // No-op
}

void DenseMatrixView::zeroVectorEntries(std::span<const GlobalIndex> /*dofs*/)
{
    // No-op
}

void DenseMatrixView::beginAssemblyPhase()
{
    phase_ = AssemblyPhase::Building;
}

void DenseMatrixView::endAssemblyPhase()
{
    phase_ = AssemblyPhase::Flushing;
}

void DenseMatrixView::finalizeAssembly()
{
    phase_ = AssemblyPhase::Finalized;
}

AssemblyPhase DenseMatrixView::getPhase() const noexcept
{
    return phase_;
}

Real DenseMatrixView::operator()(GlobalIndex row, GlobalIndex col) const
{
    if (row < 0 || row >= n_rows_ || col < 0 || col >= n_cols_) {
        throw std::out_of_range("DenseMatrixView: index out of range");
    }
    return data_[static_cast<std::size_t>(row * n_cols_ + col)];
}

void DenseMatrixView::clear()
{
    std::fill(data_.begin(), data_.end(), 0.0);
    phase_ = AssemblyPhase::NotStarted;
}

void DenseMatrixView::zero()
{
    std::fill(data_.begin(), data_.end(), 0.0);
}

Real DenseMatrixView::getMatrixEntry(GlobalIndex row, GlobalIndex col) const
{
    if (row < 0 || row >= n_rows_ || col < 0 || col >= n_cols_) {
        return 0.0;
    }
    return data_[static_cast<std::size_t>(row * n_cols_ + col)];
}

bool DenseMatrixView::isSymmetric(Real tol) const
{
    if (n_rows_ != n_cols_) return false;

    for (GlobalIndex i = 0; i < n_rows_; ++i) {
        for (GlobalIndex j = i + 1; j < n_cols_; ++j) {
            const auto ij = static_cast<std::size_t>(i * n_cols_ + j);
            const auto ji = static_cast<std::size_t>(j * n_cols_ + i);
            if (std::abs(data_[ij] - data_[ji]) > tol) {
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// DenseVectorView Implementation
// ============================================================================

DenseVectorView::DenseVectorView(GlobalIndex size)
    : size_(size)
    , data_(static_cast<std::size_t>(size), 0.0)
{
    if (size < 0) {
        throw std::invalid_argument("DenseVectorView: negative size");
    }
}

// Matrix operations (no-op for vector-only)
void DenseVectorView::addMatrixEntries(
    std::span<const GlobalIndex> /*dofs*/,
    std::span<const Real> /*local_matrix*/,
    AddMode /*mode*/)
{
    // No-op
}

void DenseVectorView::addMatrixEntries(
    std::span<const GlobalIndex> /*row_dofs*/,
    std::span<const GlobalIndex> /*col_dofs*/,
    std::span<const Real> /*local_matrix*/,
    AddMode /*mode*/)
{
    // No-op
}

void DenseVectorView::addMatrixEntry(
    GlobalIndex /*row*/,
    GlobalIndex /*col*/,
    Real /*value*/,
    AddMode /*mode*/)
{
    // No-op
}

void DenseVectorView::setDiagonal(
    std::span<const GlobalIndex> /*dofs*/,
    std::span<const Real> /*values*/)
{
    // No-op
}

void DenseVectorView::setDiagonal(GlobalIndex /*dof*/, Real /*value*/)
{
    // No-op
}

void DenseVectorView::zeroRows(
    std::span<const GlobalIndex> /*rows*/,
    bool /*set_diagonal*/)
{
    // No-op
}

void DenseVectorView::addVectorEntries(
    std::span<const GlobalIndex> dofs,
    std::span<const Real> local_vector,
    AddMode mode)
{
    if (dofs.size() != local_vector.size()) {
        throw std::invalid_argument(
            "DenseVectorView::addVectorEntries: size mismatch");
    }

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        addVectorEntry(dofs[i], local_vector[i], mode);
    }
}

void DenseVectorView::addVectorEntry(
    GlobalIndex dof,
    Real value,
    AddMode mode)
{
    if (dof < 0 || dof >= size_) {
        return;  // Silently ignore out-of-range
    }

    const auto idx = static_cast<std::size_t>(dof);

    switch (mode) {
        case AddMode::Add:
            data_[idx] += value;
            break;
        case AddMode::Insert:
            data_[idx] = value;
            break;
        case AddMode::Max:
            data_[idx] = std::max(data_[idx], value);
            break;
        case AddMode::Min:
            data_[idx] = std::min(data_[idx], value);
            break;
    }
}

void DenseVectorView::setVectorEntries(
    std::span<const GlobalIndex> dofs,
    std::span<const Real> values)
{
    if (dofs.size() != values.size()) {
        throw std::invalid_argument(
            "DenseVectorView::setVectorEntries: size mismatch");
    }

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        if (dofs[i] >= 0 && dofs[i] < size_) {
            data_[static_cast<std::size_t>(dofs[i])] = values[i];
        }
    }
}

void DenseVectorView::zeroVectorEntries(std::span<const GlobalIndex> dofs)
{
    for (const GlobalIndex dof : dofs) {
        if (dof >= 0 && dof < size_) {
            data_[static_cast<std::size_t>(dof)] = 0.0;
        }
    }
}

Real DenseVectorView::getVectorEntry(GlobalIndex dof) const
{
    if (dof < 0 || dof >= size_) {
        return 0.0;
    }
    return data_[static_cast<std::size_t>(dof)];
}

void DenseVectorView::beginAssemblyPhase()
{
    phase_ = AssemblyPhase::Building;
}

void DenseVectorView::endAssemblyPhase()
{
    phase_ = AssemblyPhase::Flushing;
}

void DenseVectorView::finalizeAssembly()
{
    phase_ = AssemblyPhase::Finalized;
}

AssemblyPhase DenseVectorView::getPhase() const noexcept
{
    return phase_;
}

Real DenseVectorView::operator[](GlobalIndex dof) const
{
    if (dof < 0 || dof >= size_) {
        throw std::out_of_range("DenseVectorView: index out of range");
    }
    return data_[static_cast<std::size_t>(dof)];
}

void DenseVectorView::clear()
{
    std::fill(data_.begin(), data_.end(), 0.0);
    phase_ = AssemblyPhase::NotStarted;
}

void DenseVectorView::zero()
{
    std::fill(data_.begin(), data_.end(), 0.0);
}

Real DenseVectorView::norm() const
{
    Real sum = 0.0;
    for (const Real val : data_) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

// ============================================================================
// DenseSystemView Implementation
// ============================================================================

DenseSystemView::DenseSystemView(GlobalIndex n_dofs)
    : n_rows_(n_dofs)
    , n_cols_(n_dofs)
    , matrix_data_(static_cast<std::size_t>(n_dofs * n_dofs), 0.0)
    , vector_data_(static_cast<std::size_t>(n_dofs), 0.0)
{
    if (n_dofs < 0) {
        throw std::invalid_argument("DenseSystemView: negative size");
    }
}

DenseSystemView::DenseSystemView(GlobalIndex n_rows, GlobalIndex n_cols)
    : n_rows_(n_rows)
    , n_cols_(n_cols)
    , matrix_data_(static_cast<std::size_t>(n_rows * n_cols), 0.0)
    , vector_data_(static_cast<std::size_t>(n_rows), 0.0)
{
    if (n_rows < 0 || n_cols < 0) {
        throw std::invalid_argument("DenseSystemView: negative dimensions");
    }
}

void DenseSystemView::addMatrixEntries(
    std::span<const GlobalIndex> dofs,
    std::span<const Real> local_matrix,
    AddMode mode)
{
    addMatrixEntries(dofs, dofs, local_matrix, mode);
}

void DenseSystemView::addMatrixEntries(
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    std::span<const Real> local_matrix,
    AddMode mode)
{
    const auto n_rows = static_cast<GlobalIndex>(row_dofs.size());
    const auto n_cols = static_cast<GlobalIndex>(col_dofs.size());

    if (local_matrix.size() != static_cast<std::size_t>(n_rows * n_cols)) {
        throw std::invalid_argument(
            "DenseSystemView::addMatrixEntries: local_matrix size mismatch");
    }

    for (GlobalIndex i = 0; i < n_rows; ++i) {
        const GlobalIndex row = row_dofs[static_cast<std::size_t>(i)];
        if (row < 0 || row >= n_rows_) continue;

        for (GlobalIndex j = 0; j < n_cols; ++j) {
            const GlobalIndex col = col_dofs[static_cast<std::size_t>(j)];
            if (col < 0 || col >= n_cols_) continue;

            const auto local_idx = static_cast<std::size_t>(i * n_cols + j);
            const auto global_idx = static_cast<std::size_t>(row * n_cols_ + col);
            const Real value = local_matrix[local_idx];

            switch (mode) {
                case AddMode::Add:
                    matrix_data_[global_idx] += value;
                    break;
                case AddMode::Insert:
                    matrix_data_[global_idx] = value;
                    break;
                case AddMode::Max:
                    matrix_data_[global_idx] = std::max(matrix_data_[global_idx], value);
                    break;
                case AddMode::Min:
                    matrix_data_[global_idx] = std::min(matrix_data_[global_idx], value);
                    break;
            }
        }
    }
}

void DenseSystemView::addMatrixEntry(
    GlobalIndex row,
    GlobalIndex col,
    Real value,
    AddMode mode)
{
    if (row < 0 || row >= n_rows_ || col < 0 || col >= n_cols_) {
        return;
    }

    const auto idx = static_cast<std::size_t>(row * n_cols_ + col);

    switch (mode) {
        case AddMode::Add:
            matrix_data_[idx] += value;
            break;
        case AddMode::Insert:
            matrix_data_[idx] = value;
            break;
        case AddMode::Max:
            matrix_data_[idx] = std::max(matrix_data_[idx], value);
            break;
        case AddMode::Min:
            matrix_data_[idx] = std::min(matrix_data_[idx], value);
            break;
    }
}

void DenseSystemView::setDiagonal(
    std::span<const GlobalIndex> dofs,
    std::span<const Real> values)
{
    if (dofs.size() != values.size()) {
        throw std::invalid_argument("DenseSystemView::setDiagonal: size mismatch");
    }

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        setDiagonal(dofs[i], values[i]);
    }
}

void DenseSystemView::setDiagonal(GlobalIndex dof, Real value)
{
    if (dof < 0 || dof >= std::min(n_rows_, n_cols_)) {
        return;
    }
    const auto idx = static_cast<std::size_t>(dof * n_cols_ + dof);
    matrix_data_[idx] = value;
}

void DenseSystemView::zeroRows(
    std::span<const GlobalIndex> rows,
    bool set_diagonal)
{
    for (const GlobalIndex row : rows) {
        if (row < 0 || row >= n_rows_) continue;

        for (GlobalIndex col = 0; col < n_cols_; ++col) {
            const auto idx = static_cast<std::size_t>(row * n_cols_ + col);
            matrix_data_[idx] = 0.0;
        }

        if (set_diagonal && row < n_cols_) {
            const auto diag_idx = static_cast<std::size_t>(row * n_cols_ + row);
            matrix_data_[diag_idx] = 1.0;
        }
    }
}

void DenseSystemView::addVectorEntries(
    std::span<const GlobalIndex> dofs,
    std::span<const Real> local_vector,
    AddMode mode)
{
    if (dofs.size() != local_vector.size()) {
        throw std::invalid_argument(
            "DenseSystemView::addVectorEntries: size mismatch");
    }

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        addVectorEntry(dofs[i], local_vector[i], mode);
    }
}

void DenseSystemView::addVectorEntry(
    GlobalIndex dof,
    Real value,
    AddMode mode)
{
    if (dof < 0 || dof >= n_rows_) {
        return;
    }

    const auto idx = static_cast<std::size_t>(dof);

    switch (mode) {
        case AddMode::Add:
            vector_data_[idx] += value;
            break;
        case AddMode::Insert:
            vector_data_[idx] = value;
            break;
        case AddMode::Max:
            vector_data_[idx] = std::max(vector_data_[idx], value);
            break;
        case AddMode::Min:
            vector_data_[idx] = std::min(vector_data_[idx], value);
            break;
    }
}

void DenseSystemView::setVectorEntries(
    std::span<const GlobalIndex> dofs,
    std::span<const Real> values)
{
    if (dofs.size() != values.size()) {
        throw std::invalid_argument(
            "DenseSystemView::setVectorEntries: size mismatch");
    }

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        if (dofs[i] >= 0 && dofs[i] < n_rows_) {
            vector_data_[static_cast<std::size_t>(dofs[i])] = values[i];
        }
    }
}

void DenseSystemView::zeroVectorEntries(std::span<const GlobalIndex> dofs)
{
    for (const GlobalIndex dof : dofs) {
        if (dof >= 0 && dof < n_rows_) {
            vector_data_[static_cast<std::size_t>(dof)] = 0.0;
        }
    }
}

Real DenseSystemView::getVectorEntry(GlobalIndex dof) const
{
    if (dof < 0 || dof >= n_rows_) {
        return 0.0;
    }
    return vector_data_[static_cast<std::size_t>(dof)];
}

void DenseSystemView::beginAssemblyPhase()
{
    phase_ = AssemblyPhase::Building;
}

void DenseSystemView::endAssemblyPhase()
{
    phase_ = AssemblyPhase::Flushing;
}

void DenseSystemView::finalizeAssembly()
{
    phase_ = AssemblyPhase::Finalized;
}

AssemblyPhase DenseSystemView::getPhase() const noexcept
{
    return phase_;
}

Real DenseSystemView::matrixEntry(GlobalIndex row, GlobalIndex col) const
{
    if (row < 0 || row >= n_rows_ || col < 0 || col >= n_cols_) {
        throw std::out_of_range("DenseSystemView: matrix index out of range");
    }
    return matrix_data_[static_cast<std::size_t>(row * n_cols_ + col)];
}

Real DenseSystemView::vectorEntry(GlobalIndex dof) const
{
    if (dof < 0 || dof >= n_rows_) {
        throw std::out_of_range("DenseSystemView: vector index out of range");
    }
    return vector_data_[static_cast<std::size_t>(dof)];
}

void DenseSystemView::clear()
{
    std::fill(matrix_data_.begin(), matrix_data_.end(), 0.0);
    std::fill(vector_data_.begin(), vector_data_.end(), 0.0);
    phase_ = AssemblyPhase::NotStarted;
}

void DenseSystemView::zero()
{
    std::fill(matrix_data_.begin(), matrix_data_.end(), 0.0);
    std::fill(vector_data_.begin(), vector_data_.end(), 0.0);
}

Real DenseSystemView::getMatrixEntry(GlobalIndex row, GlobalIndex col) const
{
    if (row < 0 || row >= n_rows_ || col < 0 || col >= n_cols_) {
        return 0.0;
    }
    return matrix_data_[static_cast<std::size_t>(row * n_cols_ + col)];
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<GlobalSystemView> createDenseMatrixView(GlobalIndex n_rows, GlobalIndex n_cols)
{
    if (n_cols < 0) {
        n_cols = n_rows;
    }
    return std::make_unique<DenseMatrixView>(n_rows, n_cols);
}

std::unique_ptr<GlobalSystemView> createDenseVectorView(GlobalIndex size)
{
    return std::make_unique<DenseVectorView>(size);
}

std::unique_ptr<GlobalSystemView> createDenseSystemView(GlobalIndex n_rows, GlobalIndex n_cols)
{
    if (n_cols < 0) {
        n_cols = n_rows;
    }
    return std::make_unique<DenseSystemView>(n_rows, n_cols);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
