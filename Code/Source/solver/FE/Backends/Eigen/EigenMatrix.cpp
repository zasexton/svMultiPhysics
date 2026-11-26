/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Eigen/EigenMatrix.h"

#include "Backends/Eigen/EigenVector.h"
#include "Core/FEException.h"
#include "Sparsity/SparsityPattern.h"
#include "Core/EnvironmentNames.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <vector>
#include <limits>

namespace svmp {
namespace FE {
namespace backends {

#if defined(FE_HAS_EIGEN)

class EigenMatrixView final : public assembly::GlobalSystemView {
public:
    explicit EigenMatrixView(EigenMatrix& matrix) : matrix_(&matrix) {}

    void addMatrixEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_matrix,
                          assembly::AddMode mode) override
    {
        addMatrixEntries(dofs, dofs, local_matrix, mode);
    }

    void addMatrixEntries(std::span<const GlobalIndex> row_dofs,
                          std::span<const GlobalIndex> col_dofs,
                          std::span<const Real> local_matrix,
                          assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(matrix_, "EigenMatrixView::matrix");
        const GlobalIndex n_rows = static_cast<GlobalIndex>(row_dofs.size());
        const GlobalIndex n_cols = static_cast<GlobalIndex>(col_dofs.size());

        if (local_matrix.size() != static_cast<std::size_t>(n_rows * n_cols)) {
            FE_THROW(InvalidArgumentException, "EigenMatrixView::addMatrixEntries: local_matrix size mismatch");
        }

        std::vector<EigenMatrix::StorageIndex> slots(local_matrix.size());
        std::size_t k = 0;
        for (const auto row : row_dofs) {
            for (const auto col : col_dofs) {
                if (!matrix_->resolveAssemblySlot(row, col, local_matrix[k], slots[k])) return;
                ++k;
            }
        }
        for (std::size_t i = 0; i < slots.size(); ++i) {
            matrix_->writeAssemblySlot(slots[i], local_matrix[i], mode);
        }
    }

    void addMatrixEntry(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(matrix_, "EigenMatrixView::matrix");
        matrix_->addValue(row, col, value, mode);
    }

    void setDiagonal(std::span<const GlobalIndex> dofs,
                     std::span<const Real> values) override
    {
        if (dofs.size() != values.size()) {
            FE_THROW(InvalidArgumentException, "EigenMatrixView::setDiagonal: size mismatch");
        }
        std::vector<EigenMatrix::StorageIndex> slots(dofs.size());
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            if (!matrix_->resolveAssemblySlot(dofs[i], dofs[i], values[i], slots[i])) return;
        }
        for (std::size_t i = 0; i < slots.size(); ++i) {
            matrix_->writeAssemblySlot(slots[i], values[i], assembly::AddMode::Insert);
        }
    }

    void setDiagonal(GlobalIndex dof, Real value) override
    {
        addMatrixEntry(dof, dof, value, assembly::AddMode::Insert);
    }

    void zeroRows(std::span<const GlobalIndex> rows, bool set_diagonal) override
    {
        FE_CHECK_NOT_NULL(matrix_, "EigenMatrixView::matrix");
        std::vector<EigenMatrix::StorageIndex> diagonals;
        if (set_diagonal) {
            diagonals.resize(rows.size(), -1);
            for (std::size_t i = 0; i < rows.size(); ++i) {
                // A rectangular row beyond the column range has no diagonal.
                if (rows[i] >= 0 && rows[i] < matrix_->numRows() && rows[i] >= matrix_->numCols()) continue;
                if (!matrix_->resolveAssemblySlot(rows[i], rows[i], 1.0, diagonals[i])) return;
            }
        }
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto row_g = rows[i];
            if (row_g < 0 || row_g >= matrix_->numRows()) continue;
            const auto row = static_cast<EigenMatrix::StorageIndex>(row_g);
            auto& A = matrix_->eigen();
            const auto* outer = A.outerIndexPtr();
            auto* values = A.valuePtr();
            const auto start = outer[row];
            const auto end = outer[row + 1];
            for (auto k = start; k < end; ++k) {
                values[k] = 0.0;
            }
            if (set_diagonal) matrix_->writeAssemblySlot(diagonals[i], 1.0, assembly::AddMode::Insert);
        }
    }

    // Vector ops (no-op)
    void addVectorEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
    void addVectorEntry(GlobalIndex, Real, assembly::AddMode) override {}
    void setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) override {}
    void zeroVectorEntries(std::span<const GlobalIndex>) override {}

    void beginAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Building; }
    void endAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Flushing; }
    void finalizeAssembly() override
    {
        matrix_->finalizeAssembly();
        phase_ = assembly::AssemblyPhase::Finalized;
    }
    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return phase_; }

    [[nodiscard]] bool hasMatrix() const noexcept override { return true; }
    [[nodiscard]] bool hasVector() const noexcept override { return false; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return matrix_ ? matrix_->numRows() : 0; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return matrix_ ? matrix_->numCols() : 0; }
    [[nodiscard]] std::string backendName() const override { return "EigenMatrix"; }

    void zero() override
    {
        FE_CHECK_NOT_NULL(matrix_, "EigenMatrixView::matrix");
        matrix_->zero();
    }

    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override
    {
        FE_CHECK_NOT_NULL(matrix_, "EigenMatrixView::matrix");
        return matrix_->getEntry(row, col);
    }

private:
    EigenMatrix* matrix_{nullptr};
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
};

EigenMatrix::EigenMatrix(const sparsity::SparsityPattern& pattern)
{
    initializeFromPattern(pattern);
}

bool EigenMatrix::reinitFromPattern(const sparsity::SparsityPattern& pattern)
{
    initializeFromPattern(pattern);
    return true;
}

void EigenMatrix::initializeFromPattern(const sparsity::SparsityPattern& pattern)
{
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException,
                "EigenMatrix: sparsity pattern must be finalized");

    n_rows_ = pattern.numRows();
    n_cols_ = pattern.numCols();

    FE_THROW_IF(n_rows_ < 0 || n_cols_ < 0, InvalidArgumentException, "EigenMatrix: negative dimensions");
    FE_THROW_IF(n_rows_ > static_cast<GlobalIndex>(std::numeric_limits<StorageIndex>::max()) ||
                    n_cols_ > static_cast<GlobalIndex>(std::numeric_limits<StorageIndex>::max()),
                InvalidArgumentException,
                "EigenMatrix: dimensions exceed Eigen storage index range (Eigen backend is intended for small problems)");

    mat_.resize(static_cast<StorageIndex>(n_rows_), static_cast<StorageIndex>(n_cols_));

    const auto row_ptr = pattern.getRowPtr();
    const auto col_idx = pattern.getColIndices();
    FE_THROW_IF(row_ptr.size() != static_cast<std::size_t>(n_rows_ + 1), InvalidArgumentException,
                "EigenMatrix: invalid CSR row_ptr size");

    const auto nnz = row_ptr.back();
    FE_THROW_IF(nnz < 0, InvalidArgumentException, "EigenMatrix: negative nnz");
    FE_THROW_IF(nnz > static_cast<GlobalIndex>(std::numeric_limits<StorageIndex>::max()),
                InvalidArgumentException, "EigenMatrix: nnz exceeds storage index range");

    std::vector<Eigen::Triplet<Real, StorageIndex>> triplets;
    triplets.reserve(static_cast<std::size_t>(nnz));

    for (GlobalIndex r = 0; r < n_rows_; ++r) {
        const auto start = row_ptr[static_cast<std::size_t>(r)];
        const auto end = row_ptr[static_cast<std::size_t>(r + 1)];
        FE_THROW_IF(end < start, InvalidArgumentException, "EigenMatrix: invalid CSR row pointers");

        for (GlobalIndex k = start; k < end; ++k) {
            const auto c = col_idx[static_cast<std::size_t>(k)];
            if (c < 0 || c >= n_cols_) {
                FE_THROW(InvalidArgumentException, "EigenMatrix: invalid column index in sparsity pattern");
            }
            triplets.emplace_back(static_cast<StorageIndex>(r), static_cast<StorageIndex>(c), 0.0);
        }
    }

    mat_.setFromTriplets(triplets.begin(), triplets.end());
    mat_.makeCompressed();
    assembly_error_.reset();
    FE_THROW_IF(!mat_.isCompressed(), FEException, "EigenMatrix: failed to compress sparse matrix");
}

void EigenMatrix::zero()
{
    assembly_error_.reset();
    if (mat_.nonZeros()) std::fill_n(mat_.valuePtr(), mat_.nonZeros(), Real(0));
}

void EigenMatrix::finalizeAssembly()
{
    assembly_error_.check("Eigen");
}

void EigenMatrix::mult(const GenericVector& x, GenericVector& y) const
{
    const auto* x_e = dynamic_cast<const EigenVector*>(&x);
    auto* y_e = dynamic_cast<EigenVector*>(&y);

    FE_THROW_IF(!x_e || !y_e, InvalidArgumentException, "EigenMatrix::mult: backend mismatch");
    FE_THROW_IF(x_e->eigen().size() != mat_.cols() || y_e->eigen().size() != mat_.rows(),
                InvalidArgumentException, "EigenMatrix::mult: size mismatch");

    y_e->eigen().noalias() = mat_ * x_e->eigen();
}

void EigenMatrix::multAdd(const GenericVector& x, GenericVector& y) const
{
    const auto* x_e = dynamic_cast<const EigenVector*>(&x);
    auto* y_e = dynamic_cast<EigenVector*>(&y);

    FE_THROW_IF(!x_e || !y_e, InvalidArgumentException, "EigenMatrix::multAdd: backend mismatch");
    FE_THROW_IF(x_e->eigen().size() != mat_.cols() || y_e->eigen().size() != mat_.rows(),
                InvalidArgumentException, "EigenMatrix::multAdd: size mismatch");

    y_e->eigen().noalias() += mat_ * x_e->eigen();
}

std::unique_ptr<assembly::GlobalSystemView> EigenMatrix::createAssemblyView()
{
    return std::make_unique<EigenMatrixView>(*this);
}

Real EigenMatrix::getEntry(GlobalIndex row, GlobalIndex col) const
{
    if (row < 0 || row >= n_rows_ || col < 0 || col >= n_cols_) {
        return 0.0;
    }
    return mat_.coeff(static_cast<StorageIndex>(row), static_cast<StorageIndex>(col));
}

void EigenMatrix::addValue(GlobalIndex row_g, GlobalIndex col_g, Real value, assembly::AddMode mode)
{
    StorageIndex slot;
    if (resolveAssemblySlot(row_g, col_g, value, slot)) writeAssemblySlot(slot, value, mode);
}

bool EigenMatrix::resolveAssemblySlot(GlobalIndex row, GlobalIndex col, Real value, StorageIndex& slot)
{
    slot = -1;
    if (row >= 0 && row < n_rows_ && col >= 0 && col < n_cols_) {
        const auto* outer = mat_.outerIndexPtr();
        const auto start = outer[row];
        const auto end = outer[row + 1];
        if (start != end) {
            const auto* inner = mat_.innerIndexPtr();
            const auto* found = std::lower_bound(inner + start, inner + end, static_cast<StorageIndex>(col));
            if (found != inner + end && *found == col) {
                slot = static_cast<StorageIndex>(found - inner);
                return true;
            }
        }
    }

    static const bool strict_pattern = [] {
        const char* setting = environment::value("SVMP_EIGEN_STRICT_PATTERN");
        return setting && *setting && *setting != '0';
    }();
    if (value != Real(0) || strict_pattern) {
        // Eigen owns one serial matrix. Defer OpenMP failures until its workers join.
        assembly_error_.reject("Eigen", row, col, value, false);
        return false;
    }
    static const bool trace_zeros = [] {
        const char* setting = environment::value("SVMP_EIGEN_COUNT_DROPPED");
        return setting && *setting && *setting != '0';
    }();
    if (trace_zeros) {
        static std::atomic<long> count{0};
        const auto n = ++count;
        if (n <= 32 || n % 100000 == 0) {
            std::fprintf(stderr, "[EigenMatrix] skipped structural zero #%ld (row=%lld col=%lld)\n",
                         n, static_cast<long long>(row), static_cast<long long>(col));
        }
    }
    return true;
}

void EigenMatrix::writeAssemblySlot(StorageIndex idx, Real value, assembly::AddMode mode)
{
    if (idx < 0) return;
    auto* values = mat_.valuePtr();
    switch (mode) {
        case assembly::AddMode::Add:
            values[idx] += value;
            break;
        case assembly::AddMode::Insert:
            values[idx] = value;
            break;
        case assembly::AddMode::Max:
            values[idx] = std::max(values[idx], value);
            break;
        case assembly::AddMode::Min:
            values[idx] = std::min(values[idx], value);
            break;
    }
}

#endif // FE_HAS_EIGEN

} // namespace backends
} // namespace FE
} // namespace svmp
