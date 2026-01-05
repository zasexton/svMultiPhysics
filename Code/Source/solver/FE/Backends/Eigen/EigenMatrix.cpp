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

#include <algorithm>
#include <limits>

namespace svmp {
namespace FE {
namespace backends {

#if defined(FE_HAS_EIGEN)

namespace {

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

        for (GlobalIndex i = 0; i < n_rows; ++i) {
            const GlobalIndex row = row_dofs[static_cast<std::size_t>(i)];
            if (row < 0 || row >= matrix_->numRows()) continue;

            for (GlobalIndex j = 0; j < n_cols; ++j) {
                const GlobalIndex col = col_dofs[static_cast<std::size_t>(j)];
                if (col < 0 || col >= matrix_->numCols()) continue;

                const auto local_idx = static_cast<std::size_t>(i * n_cols + j);
                matrix_->addValue(row, col, local_matrix[local_idx], mode);
            }
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
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            setDiagonal(dofs[i], values[i]);
        }
    }

    void setDiagonal(GlobalIndex dof, Real value) override
    {
        addMatrixEntry(dof, dof, value, assembly::AddMode::Insert);
    }

    void zeroRows(std::span<const GlobalIndex> rows, bool set_diagonal) override
    {
        FE_CHECK_NOT_NULL(matrix_, "EigenMatrixView::matrix");
        for (const GlobalIndex row_g : rows) {
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
            if (set_diagonal && row_g < matrix_->numCols()) {
                matrix_->addValue(row_g, row_g, 1.0, assembly::AddMode::Insert);
            }
        }
    }

    // Vector ops (no-op)
    void addVectorEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
    void addVectorEntry(GlobalIndex, Real, assembly::AddMode) override {}
    void setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) override {}
    void zeroVectorEntries(std::span<const GlobalIndex>) override {}

    void beginAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Building; }
    void endAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Flushing; }
    void finalizeAssembly() override { phase_ = assembly::AssemblyPhase::Finalized; }
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

} // namespace

EigenMatrix::EigenMatrix(const sparsity::SparsityPattern& pattern)
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
    FE_THROW_IF(!mat_.isCompressed(), FEException, "EigenMatrix: failed to compress sparse matrix");
}

void EigenMatrix::zero()
{
    std::fill(mat_.valuePtr(), mat_.valuePtr() + mat_.nonZeros(), 0.0);
}

void EigenMatrix::finalizeAssembly()
{
    // Eigen matrix is assembled in-place and stays compressed; nothing to do.
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
    if (row_g < 0 || row_g >= n_rows_ || col_g < 0 || col_g >= n_cols_) {
        return;
    }

    const auto row = static_cast<StorageIndex>(row_g);
    const auto col = static_cast<StorageIndex>(col_g);

    const auto* outer = mat_.outerIndexPtr();
    const auto* inner = mat_.innerIndexPtr();
    auto* values = mat_.valuePtr();

    const StorageIndex start = outer[row];
    const StorageIndex end = outer[row + 1];

    const auto* begin = inner + start;
    const auto* finish = inner + end;
    const auto it = std::lower_bound(begin, finish, col);
    if (it == finish || *it != col) {
        return;
    }

    const auto idx = static_cast<StorageIndex>(it - inner);

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
