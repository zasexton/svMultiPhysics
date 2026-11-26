/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/PETSc/PetscMatrix.h"

#if defined(FE_HAS_PETSC)

#include "Backends/PETSc/PetscVector.h"
#include "Core/FEException.h"
#include "Sparsity/DistributedSparsityPattern.h"
#include "Sparsity/SparsityPattern.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

namespace {

[[nodiscard]] PetscInt asPetscInt(GlobalIndex v, const char* what)
{
    FE_THROW_IF(v < 0, InvalidArgumentException, std::string("PETSc: negative ") + what);
    FE_THROW_IF(v > static_cast<GlobalIndex>(std::numeric_limits<PetscInt>::max()),
                InvalidArgumentException,
                std::string("PETSc: ") + what + " exceeds PetscInt range");
    return static_cast<PetscInt>(v);
}

// AIJ preallocation from row counts reserves capacity but does not install
// column indices. Copy the actual graph before locking its structure.
void createFromCsr(Mat* matrix, GlobalIndex local_rows, GlobalIndex local_cols,
                   GlobalIndex global_rows, GlobalIndex global_cols,
                   const std::vector<PetscInt>& offsets, const std::vector<PetscInt>& columns)
{
    FE_PETSC_CALL(MatCreate(PETSC_COMM_WORLD, matrix));
    try {
        FE_PETSC_CALL(MatSetSizes(*matrix, asPetscInt(local_rows, "local rows"), asPetscInt(local_cols, "local cols"),
                                  asPetscInt(global_rows, "global rows"), asPetscInt(global_cols, "global cols")));
        FE_PETSC_CALL(MatSetType(*matrix, MATAIJ));
        FE_PETSC_CALL(MatSetFromOptions(*matrix));
        FE_PETSC_CALL(MatSeqAIJSetPreallocationCSR(*matrix, offsets.data(), columns.data(), nullptr));
        FE_PETSC_CALL(MatMPIAIJSetPreallocationCSR(*matrix, offsets.data(), columns.data(), nullptr));
        FE_PETSC_CALL(MatSetUp(*matrix));
        FE_PETSC_CALL(MatSetOption(*matrix, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
    } catch (...) {
        MatDestroy(matrix);
        throw;
    }
}

InsertMode toPetscInsertMode(assembly::AddMode mode)
{
    switch (mode) {
        case assembly::AddMode::Add: return ADD_VALUES;
        case assembly::AddMode::Insert: return INSERT_VALUES;
        case assembly::AddMode::Max: return MAX_VALUES;
        case assembly::AddMode::Min: return MIN_VALUES;
        default: return ADD_VALUES;
    }
}

class PetscMatrixView final : public assembly::GlobalSystemView {
public:
    explicit PetscMatrixView(PetscMatrix& matrix) : matrix_(&matrix) {}

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
        FE_CHECK_NOT_NULL(matrix_, "PetscMatrixView::matrix");
        const GlobalIndex n_rows = static_cast<GlobalIndex>(row_dofs.size());
        const GlobalIndex n_cols = static_cast<GlobalIndex>(col_dofs.size());

        if (local_matrix.size() != static_cast<std::size_t>(n_rows * n_cols)) {
            FE_THROW(InvalidArgumentException, "PetscMatrixView::addMatrixEntries: local_matrix size mismatch");
        }

        bool all_valid = true;
        for (const auto r : row_dofs) {
            all_valid = all_valid && (r >= 0) && (r < matrix_->numRows());
        }
        for (const auto c : col_dofs) {
            all_valid = all_valid && (c >= 0) && (c < matrix_->numCols());
        }

        if (!all_valid) {
            // Conservative fallback matching Eigen/FSILS behavior.
            for (GlobalIndex i = 0; i < n_rows; ++i) {
                const GlobalIndex row = row_dofs[static_cast<std::size_t>(i)];
                if (row < 0 || row >= matrix_->numRows()) continue;

                for (GlobalIndex j = 0; j < n_cols; ++j) {
                    const GlobalIndex col = col_dofs[static_cast<std::size_t>(j)];
                    if (col < 0 || col >= matrix_->numCols()) continue;

                    const auto local_idx = static_cast<std::size_t>(i * n_cols + j);
                    addMatrixEntry(row, col, local_matrix[local_idx], mode);
                }
            }
            return;
        }

        std::vector<PetscInt> rows(row_dofs.size());
        std::vector<PetscInt> cols(col_dofs.size());
        for (std::size_t i = 0; i < row_dofs.size(); ++i) {
            rows[i] = static_cast<PetscInt>(row_dofs[i]);
        }
        for (std::size_t j = 0; j < col_dofs.size(); ++j) {
            cols[j] = static_cast<PetscInt>(col_dofs[j]);
        }

        std::vector<PetscScalar> vals(local_matrix.size());
        for (std::size_t k = 0; k < local_matrix.size(); ++k) {
            vals[k] = static_cast<PetscScalar>(local_matrix[k]);
        }

        FE_PETSC_CALL(MatSetValues(matrix_->petsc(),
                                   static_cast<PetscInt>(rows.size()),
                                   rows.data(),
                                   static_cast<PetscInt>(cols.size()),
                                   cols.data(),
                                   vals.data(),
                                   toPetscInsertMode(mode)));
    }

    void addMatrixEntry(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(matrix_, "PetscMatrixView::matrix");
        if (row < 0 || row >= matrix_->numRows() || col < 0 || col >= matrix_->numCols()) {
            return;
        }
        const PetscInt r = static_cast<PetscInt>(row);
        const PetscInt c = static_cast<PetscInt>(col);
        const PetscScalar v = static_cast<PetscScalar>(value);
        FE_PETSC_CALL(MatSetValues(matrix_->petsc(), 1, &r, 1, &c, &v, toPetscInsertMode(mode)));
    }

    void setDiagonal(std::span<const GlobalIndex> dofs,
                     std::span<const Real> values) override
    {
        if (dofs.size() != values.size()) {
            FE_THROW(InvalidArgumentException, "PetscMatrixView::setDiagonal: size mismatch");
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
        FE_CHECK_NOT_NULL(matrix_, "PetscMatrixView::matrix");
        std::vector<PetscInt> valid;
        valid.reserve(rows.size());
        for (const auto r : rows) {
            if (r < 0 || r >= matrix_->numRows()) continue;
            valid.push_back(static_cast<PetscInt>(r));
        }
        const PetscScalar diag = set_diagonal ? static_cast<PetscScalar>(1.0) : static_cast<PetscScalar>(0.0);
        // MatZeroRows is collective. Some diagnostic constraints are
        // rank-local, so ranks with no locally valid rows must still enter the
        // call with a zero count and a null row pointer.
        FE_PETSC_CALL(MatZeroRows(matrix_->petsc(),
                                  static_cast<PetscInt>(valid.size()),
                                  valid.empty() ? nullptr : valid.data(),
                                  diag,
                                  nullptr,
                                  nullptr));
    }

    // Vector ops (no-op)
    void addVectorEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
    void addVectorEntry(GlobalIndex, Real, assembly::AddMode) override {}
    void setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) override {}
    void zeroVectorEntries(std::span<const GlobalIndex>) override {}

    void beginAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Building; }

    void endAssemblyPhase() override
    {
        FE_CHECK_NOT_NULL(matrix_, "PetscMatrixView::matrix");
        FE_PETSC_CALL(MatAssemblyBegin(matrix_->petsc(), MAT_FLUSH_ASSEMBLY));
        FE_PETSC_CALL(MatAssemblyEnd(matrix_->petsc(), MAT_FLUSH_ASSEMBLY));
        phase_ = assembly::AssemblyPhase::Flushing;
    }

    void finalizeAssembly() override
    {
        FE_CHECK_NOT_NULL(matrix_, "PetscMatrixView::matrix");
        FE_PETSC_CALL(MatAssemblyBegin(matrix_->petsc(), MAT_FINAL_ASSEMBLY));
        FE_PETSC_CALL(MatAssemblyEnd(matrix_->petsc(), MAT_FINAL_ASSEMBLY));
        phase_ = assembly::AssemblyPhase::Finalized;
    }

    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return phase_; }

    [[nodiscard]] bool hasMatrix() const noexcept override { return true; }
    [[nodiscard]] bool hasVector() const noexcept override { return false; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return matrix_ ? matrix_->numRows() : 0; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return matrix_ ? matrix_->numCols() : 0; }
    [[nodiscard]] bool isDistributed() const noexcept override { return true; }
    [[nodiscard]] std::string backendName() const override { return "PETScMatrix"; }

    void zero() override
    {
        FE_CHECK_NOT_NULL(matrix_, "PetscMatrixView::matrix");
        matrix_->zero();
    }

    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override
    {
        FE_CHECK_NOT_NULL(matrix_, "PetscMatrixView::matrix");
        return matrix_->getEntry(row, col);
    }

private:
    PetscMatrix* matrix_{nullptr};
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
};

} // namespace

PetscMatrix::PetscMatrix(const sparsity::SparsityPattern& sparsity)
{
    allocateFromSequential(sparsity);
}

PetscMatrix::PetscMatrix(const sparsity::DistributedSparsityPattern& sparsity)
{
    allocateFromDistributed(sparsity);
}

PetscMatrix::~PetscMatrix()
{
    if (work_) {
        FE_PETSC_CALL(VecDestroy(&work_));
    }
    if (mat_) {
        FE_PETSC_CALL(MatDestroy(&mat_));
    }
}

PetscMatrix::PetscMatrix(PetscMatrix&& other) noexcept
{
    *this = std::move(other);
}

PetscMatrix& PetscMatrix::operator=(PetscMatrix&& other) noexcept
{
    if (this == &other) {
        return *this;
    }
    if (mat_) {
        MatDestroy(&mat_);
    }
    if (work_) {
        VecDestroy(&work_);
    }
    mat_ = other.mat_;
    work_ = other.work_;
    other.mat_ = nullptr;
    other.work_ = nullptr;
    return *this;
}

void PetscMatrix::allocateFromSequential(const sparsity::SparsityPattern& pattern)
{
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException, "PetscMatrix: sparsity pattern must be finalized");

    PetscMPIInt comm_size = 1;
    FE_PETSC_CALL(MPI_Comm_size(PETSC_COMM_WORLD, &comm_size));
    FE_THROW_IF(comm_size != 1, NotImplementedException,
                "PetscMatrix: sequential SparsityPattern is only supported in serial; "
                "use DistributedSparsityPattern for MPI runs");

    std::vector<PetscInt> offsets, columns;
    offsets.reserve(pattern.getRowPtr().size());
    columns.reserve(pattern.getColIndices().size());
    for (auto offset : pattern.getRowPtr()) offsets.push_back(asPetscInt(offset, "row offset"));
    for (auto col : pattern.getColIndices()) columns.push_back(asPetscInt(col, "column"));
    createFromCsr(&mat_, pattern.numRows(), pattern.numCols(), pattern.numRows(), pattern.numCols(), offsets, columns);
}

void PetscMatrix::allocateFromDistributed(const sparsity::DistributedSparsityPattern& dist)
{
    FE_THROW_IF(!dist.isFinalized(), InvalidArgumentException, "PetscMatrix: distributed sparsity must be finalized");
    std::vector<PetscInt> offsets{0}, columns;
    offsets.reserve(static_cast<std::size_t>(dist.numOwnedRows()) + 1);
    for (GlobalIndex row = 0; row < dist.numOwnedRows(); ++row) {
        for (auto col : dist.getRowDiagCols(row)) {
            columns.push_back(asPetscInt(dist.ownedCols().first + col, "diagonal column"));
        }
        for (auto col : dist.getRowOffdiagCols(row)) {
            columns.push_back(asPetscInt(dist.ghostColToGlobal(col), "off-diagonal column"));
        }
        offsets.push_back(asPetscInt(columns.size(), "row offset"));
    }
    createFromCsr(&mat_, dist.numOwnedRows(), dist.numOwnedCols(), dist.globalRows(), dist.globalCols(), offsets, columns);
}

GlobalIndex PetscMatrix::numRows() const noexcept
{
    if (!mat_) return 0;
    PetscInt m = 0, n = 0;
    MatGetSize(mat_, &m, &n);
    return static_cast<GlobalIndex>(m);
}

GlobalIndex PetscMatrix::numCols() const noexcept
{
    if (!mat_) return 0;
    PetscInt m = 0, n = 0;
    MatGetSize(mat_, &m, &n);
    return static_cast<GlobalIndex>(n);
}

void PetscMatrix::zero()
{
    FE_PETSC_CALL(MatZeroEntries(mat_));
}

void PetscMatrix::finalizeAssembly()
{
    FE_PETSC_CALL(MatAssemblyBegin(mat_, MAT_FINAL_ASSEMBLY));
    FE_PETSC_CALL(MatAssemblyEnd(mat_, MAT_FINAL_ASSEMBLY));
}

void PetscMatrix::mult(const GenericVector& x_in, GenericVector& y_in) const
{
    const auto* x = dynamic_cast<const PetscVector*>(&x_in);
    auto* y = dynamic_cast<PetscVector*>(&y_in);
    FE_THROW_IF(!x || !y, InvalidArgumentException, "PetscMatrix::mult: backend mismatch");
    FE_PETSC_CALL(MatMult(mat_, x->petsc(), y->petsc()));
    y->invalidateLocalCache();
}

void PetscMatrix::multAdd(const GenericVector& x_in, GenericVector& y_in) const
{
    const auto* x = dynamic_cast<const PetscVector*>(&x_in);
    auto* y = dynamic_cast<PetscVector*>(&y_in);
    FE_THROW_IF(!x || !y, InvalidArgumentException, "PetscMatrix::multAdd: backend mismatch");

    if (!work_) {
        FE_PETSC_CALL(VecDuplicate(y->petsc(), &work_));
    }

    FE_PETSC_CALL(MatMult(mat_, x->petsc(), work_));
    FE_PETSC_CALL(VecAXPY(y->petsc(), static_cast<PetscScalar>(1.0), work_));
    y->invalidateLocalCache();
}

std::unique_ptr<assembly::GlobalSystemView> PetscMatrix::createAssemblyView()
{
    return std::make_unique<PetscMatrixView>(*this);
}

Real PetscMatrix::getEntry(GlobalIndex row, GlobalIndex col) const
{
    if (!mat_) return 0.0;
    if (row < 0 || col < 0) return 0.0;
    const PetscInt r = static_cast<PetscInt>(row);
    const PetscInt c = static_cast<PetscInt>(col);
    PetscScalar v = 0.0;
    FE_PETSC_CALL(MatGetValues(mat_, 1, &r, 1, &c, &v));
    return static_cast<Real>(v);
}

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_PETSC
