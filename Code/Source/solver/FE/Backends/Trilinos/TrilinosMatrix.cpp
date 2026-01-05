/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Trilinos/TrilinosMatrix.h"

#if defined(FE_HAS_TRILINOS)

#include "Backends/Trilinos/TrilinosVector.h"
#include "Core/FEException.h"
#include "Sparsity/DistributedSparsityPattern.h"
#include "Sparsity/SparsityPattern.h"

#include <Teuchos_ArrayView.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_Transp.hpp>

#include <algorithm>
#include <limits>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

namespace {

[[nodiscard]] trilinos::GO asGo(GlobalIndex v, const char* what)
{
    FE_THROW_IF(v < 0, InvalidArgumentException, std::string("Trilinos: negative ") + what);
    FE_THROW_IF(v > static_cast<GlobalIndex>(std::numeric_limits<trilinos::GO>::max()),
                InvalidArgumentException,
                std::string("Trilinos: ") + what + " exceeds GO range");
    return static_cast<trilinos::GO>(v);
}

[[nodiscard]] trilinos::Scalar toScalar(Real v) { return static_cast<trilinos::Scalar>(v); }

class TrilinosMatrixView final : public assembly::GlobalSystemView {
public:
    explicit TrilinosMatrixView(TrilinosMatrix& matrix) : matrix_(&matrix) {}

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
        FE_CHECK_NOT_NULL(matrix_, "TrilinosMatrixView::matrix");
        const GlobalIndex n_rows = static_cast<GlobalIndex>(row_dofs.size());
        const GlobalIndex n_cols = static_cast<GlobalIndex>(col_dofs.size());

        if (local_matrix.size() != static_cast<std::size_t>(n_rows * n_cols)) {
            FE_THROW(InvalidArgumentException, "TrilinosMatrixView::addMatrixEntries: local_matrix size mismatch");
        }

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
    }

    void addMatrixEntry(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(matrix_, "TrilinosMatrixView::matrix");
        if (row < 0 || row >= matrix_->numRows() || col < 0 || col >= matrix_->numCols()) {
            return;
        }

        const auto A = matrix_->tpetra();
        const auto rowMap = A->getRowMap();
        const auto colMap = A->getColMap();

        const auto lrow = rowMap->getLocalElement(asGo(row, "row"));
        if (lrow == Teuchos::OrdinalTraits<trilinos::LO>::invalid()) {
            FE_THROW(NotImplementedException,
                     "TrilinosMatrixView::addMatrixEntry: nonlocal row insertion is not supported; "
                     "assemble owned rows only or use PETSc for off-process insertion");
        }

        const auto lcol = colMap->getLocalElement(asGo(col, "col"));
        if (lcol == Teuchos::OrdinalTraits<trilinos::LO>::invalid()) {
            return; // not in this rank's column map
        }

        const trilinos::LO col_idx[1] = {static_cast<trilinos::LO>(lcol)};
        const trilinos::Scalar val[1] = {toScalar(value)};
        const Teuchos::ArrayView<const trilinos::LO> cols(col_idx, 1);
        const Teuchos::ArrayView<const trilinos::Scalar> vals(val, 1);

        switch (mode) {
            case assembly::AddMode::Add:
                A->sumIntoLocalValues(static_cast<trilinos::LO>(lrow), cols, vals);
                break;
            case assembly::AddMode::Insert:
                A->replaceLocalValues(static_cast<trilinos::LO>(lrow), cols, vals);
                break;
            case assembly::AddMode::Max: {
                const auto cur = matrix_->getEntry(row, col);
                const auto next = std::max(cur, value);
                const trilinos::Scalar v2[1] = {toScalar(next)};
                A->replaceLocalValues(static_cast<trilinos::LO>(lrow), cols, Teuchos::ArrayView<const trilinos::Scalar>(v2, 1));
                break;
            }
            case assembly::AddMode::Min: {
                const auto cur = matrix_->getEntry(row, col);
                const auto next = std::min(cur, value);
                const trilinos::Scalar v2[1] = {toScalar(next)};
                A->replaceLocalValues(static_cast<trilinos::LO>(lrow), cols, Teuchos::ArrayView<const trilinos::Scalar>(v2, 1));
                break;
            }
        }
    }

    void setDiagonal(std::span<const GlobalIndex> dofs,
                     std::span<const Real> values) override
    {
        if (dofs.size() != values.size()) {
            FE_THROW(InvalidArgumentException, "TrilinosMatrixView::setDiagonal: size mismatch");
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
        FE_CHECK_NOT_NULL(matrix_, "TrilinosMatrixView::matrix");
        const auto A = matrix_->tpetra();
        const auto rowMap = A->getRowMap();

        for (const auto row : rows) {
            if (row < 0 || row >= matrix_->numRows()) continue;
            const auto lrow = rowMap->getLocalElement(asGo(row, "row"));
            if (lrow == Teuchos::OrdinalTraits<trilinos::LO>::invalid()) continue;

            Teuchos::ArrayView<const trilinos::LO> cols;
            Teuchos::ArrayView<const trilinos::Scalar> vals;
            A->getLocalRowView(static_cast<trilinos::LO>(lrow), cols, vals);
            if (cols.empty()) continue;

            std::vector<trilinos::Scalar> zeros(static_cast<std::size_t>(cols.size()), toScalar(0.0));
            A->replaceLocalValues(static_cast<trilinos::LO>(lrow),
                                  cols,
                                  Teuchos::ArrayView<const trilinos::Scalar>(zeros.data(), zeros.size()));

            if (set_diagonal && row < matrix_->numCols()) {
                addMatrixEntry(row, row, 1.0, assembly::AddMode::Insert);
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
    [[nodiscard]] bool isDistributed() const noexcept override { return true; }
    [[nodiscard]] std::string backendName() const override { return "TrilinosMatrix"; }

    void zero() override
    {
        FE_CHECK_NOT_NULL(matrix_, "TrilinosMatrixView::matrix");
        matrix_->zero();
    }

    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override
    {
        FE_CHECK_NOT_NULL(matrix_, "TrilinosMatrixView::matrix");
        return matrix_->getEntry(row, col);
    }

private:
    TrilinosMatrix* matrix_{nullptr};
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
};

} // namespace

TrilinosMatrix::TrilinosMatrix(const sparsity::SparsityPattern& sparsity)
{
    allocateFromSequential(sparsity);
}

TrilinosMatrix::TrilinosMatrix(const sparsity::DistributedSparsityPattern& sparsity)
{
    allocateFromDistributed(sparsity);
}

void TrilinosMatrix::allocateFromSequential(const sparsity::SparsityPattern& pattern)
{
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException, "TrilinosMatrix: sparsity pattern must be finalized");

    const auto comm = Tpetra::getDefaultComm();
    FE_THROW_IF(comm->getSize() != 1, NotImplementedException,
                "TrilinosMatrix: sequential SparsityPattern is only supported in serial; "
                "use DistributedSparsityPattern for MPI runs");
    const auto n_rows = pattern.numRows();
    const auto n_cols = pattern.numCols();

    row_map_ = Teuchos::rcp(new trilinos::Map(static_cast<Tpetra::global_size_t>(asGo(n_rows, "rows")), 0, comm));
    domain_map_ = Teuchos::rcp(new trilinos::Map(static_cast<Tpetra::global_size_t>(asGo(n_cols, "cols")), 0, comm));
    range_map_ = row_map_;

    // Over-allocate per row by the max CSR nnz count.
    const auto row_ptr = pattern.getRowPtr();
    FE_THROW_IF(row_ptr.size() != static_cast<std::size_t>(n_rows + 1), InvalidArgumentException,
                "TrilinosMatrix: invalid CSR row_ptr size");

    GlobalIndex max_row_nnz = 0;
    for (GlobalIndex r = 0; r < n_rows; ++r) {
        const auto start = row_ptr[static_cast<std::size_t>(r)];
        const auto end = row_ptr[static_cast<std::size_t>(r + 1)];
        max_row_nnz = std::max(max_row_nnz, end - start);
    }

    mat_ = Teuchos::rcp(new trilinos::CrsMatrix(row_map_, static_cast<std::size_t>(max_row_nnz)));

    const auto col_idx = pattern.getColIndices();
    for (GlobalIndex r = 0; r < n_rows; ++r) {
        const auto start = row_ptr[static_cast<std::size_t>(r)];
        const auto end = row_ptr[static_cast<std::size_t>(r + 1)];
        std::vector<trilinos::GO> cols;
        cols.reserve(static_cast<std::size_t>(end - start));
        std::vector<trilinos::Scalar> vals;
        vals.reserve(static_cast<std::size_t>(end - start));
        for (GlobalIndex k = start; k < end; ++k) {
            cols.push_back(asGo(col_idx[static_cast<std::size_t>(k)], "col"));
            vals.push_back(toScalar(0.0));
        }
        mat_->insertGlobalValues(asGo(r, "row"),
                                 Teuchos::ArrayView<const trilinos::GO>(cols.data(), cols.size()),
                                 Teuchos::ArrayView<const trilinos::Scalar>(vals.data(), vals.size()));
    }

    mat_->fillComplete(domain_map_, range_map_);
}

void TrilinosMatrix::allocateFromDistributed(const sparsity::DistributedSparsityPattern& dist)
{
    FE_THROW_IF(!dist.isFinalized(), InvalidArgumentException,
                "TrilinosMatrix: distributed sparsity pattern must be finalized");

    const auto comm = Tpetra::getDefaultComm();
    const auto n_local_rows = dist.numOwnedRows();
    const auto n_local_cols = dist.numOwnedCols();

    const auto owned_rows = dist.ownedRows();
    const auto owned_cols = dist.ownedCols();

    row_map_ = Teuchos::rcp(new trilinos::Map(static_cast<Tpetra::global_size_t>(asGo(dist.globalRows(), "global rows")),
                                              static_cast<std::size_t>(asGo(n_local_rows, "local rows")),
                                              0,
                                              comm));
    domain_map_ = Teuchos::rcp(new trilinos::Map(static_cast<Tpetra::global_size_t>(asGo(dist.globalCols(), "global cols")),
                                                 static_cast<std::size_t>(asGo(n_local_cols, "local cols")),
                                                 0,
                                                 comm));
    range_map_ = row_map_;

    // Sanity-check: this backend assumes a contiguous (block) distribution matching Tpetra::Map's default.
    if (n_local_rows > 0) {
        FE_THROW_IF(row_map_->getMinGlobalIndex() != asGo(owned_rows.first, "owned row begin") ||
                        row_map_->getMaxGlobalIndex() != asGo(owned_rows.last - 1, "owned row end"),
                    InvalidArgumentException,
                    "TrilinosMatrix: owned row range does not match Tpetra contiguous distribution; "
                    "use a block distribution compatible with Tpetra::Map(n_global, n_local, 0)");
    }
    if (n_local_cols > 0) {
        FE_THROW_IF(domain_map_->getMinGlobalIndex() != asGo(owned_cols.first, "owned col begin") ||
                        domain_map_->getMaxGlobalIndex() != asGo(owned_cols.last - 1, "owned col end"),
                    InvalidArgumentException,
                    "TrilinosMatrix: owned col range does not match Tpetra contiguous distribution; "
                    "use a block distribution compatible with Tpetra::Map(n_global, n_local, 0)");
    }

    const auto pre = dist.getPreallocationInfo();
    const GlobalIndex max_row_nnz = pre.max_diag_nnz + pre.max_offdiag_nnz;
    mat_ = Teuchos::rcp(new trilinos::CrsMatrix(row_map_, static_cast<std::size_t>(max_row_nnz)));

    for (GlobalIndex lr = 0; lr < n_local_rows; ++lr) {
        const GlobalIndex gr = owned_rows.first + lr;

        const auto diag_cols = dist.getRowDiagCols(lr);
        const auto off_cols = dist.getRowOffdiagCols(lr);

        std::vector<trilinos::GO> cols;
        cols.reserve(static_cast<std::size_t>(diag_cols.size() + off_cols.size()));
        std::vector<trilinos::Scalar> vals;
        vals.reserve(cols.size());

        for (const auto c_local : diag_cols) {
            cols.push_back(asGo(owned_cols.first + c_local, "diag col"));
            vals.push_back(toScalar(0.0));
        }
        for (const auto c_ghost : off_cols) {
            cols.push_back(asGo(dist.ghostColToGlobal(c_ghost), "offdiag col"));
            vals.push_back(toScalar(0.0));
        }

        mat_->insertGlobalValues(asGo(gr, "row"),
                                 Teuchos::ArrayView<const trilinos::GO>(cols.data(), cols.size()),
                                 Teuchos::ArrayView<const trilinos::Scalar>(vals.data(), vals.size()));
    }

    mat_->fillComplete(domain_map_, range_map_);
}

GlobalIndex TrilinosMatrix::numRows() const noexcept
{
    if (mat_.is_null()) return 0;
    return static_cast<GlobalIndex>(mat_->getGlobalNumRows());
}

GlobalIndex TrilinosMatrix::numCols() const noexcept
{
    if (mat_.is_null()) return 0;
    return static_cast<GlobalIndex>(mat_->getGlobalNumCols());
}

void TrilinosMatrix::zero()
{
    mat_->setAllToScalar(toScalar(0.0));
}

void TrilinosMatrix::finalizeAssembly()
{
    // Values are inserted directly into the fixed structure.
}

void TrilinosMatrix::mult(const GenericVector& x_in, GenericVector& y_in) const
{
    const auto* x = dynamic_cast<const TrilinosVector*>(&x_in);
    auto* y = dynamic_cast<TrilinosVector*>(&y_in);
    FE_THROW_IF(!x || !y, InvalidArgumentException, "TrilinosMatrix::mult: backend mismatch");
    mat_->apply(*x->tpetra(), *y->tpetra());
    y->invalidateLocalCache();
}

void TrilinosMatrix::multAdd(const GenericVector& x_in, GenericVector& y_in) const
{
    const auto* x = dynamic_cast<const TrilinosVector*>(&x_in);
    auto* y = dynamic_cast<TrilinosVector*>(&y_in);
    FE_THROW_IF(!x || !y, InvalidArgumentException, "TrilinosMatrix::multAdd: backend mismatch");
    mat_->apply(*x->tpetra(),
                *y->tpetra(),
                Teuchos::NO_TRANS,
                static_cast<trilinos::Scalar>(1.0),
                static_cast<trilinos::Scalar>(1.0));
    y->invalidateLocalCache();
}

std::unique_ptr<assembly::GlobalSystemView> TrilinosMatrix::createAssemblyView()
{
    return std::make_unique<TrilinosMatrixView>(*this);
}

Real TrilinosMatrix::getEntry(GlobalIndex row, GlobalIndex col) const
{
    if (row < 0 || col < 0) return 0.0;
    const auto A = mat_;
    const auto rowMap = A->getRowMap();
    const auto colMap = A->getColMap();
    const auto lrow = rowMap->getLocalElement(asGo(row, "row"));
    if (lrow == Teuchos::OrdinalTraits<trilinos::LO>::invalid()) return 0.0;
    const auto lcol = colMap->getLocalElement(asGo(col, "col"));
    if (lcol == Teuchos::OrdinalTraits<trilinos::LO>::invalid()) return 0.0;

    Teuchos::ArrayView<const trilinos::LO> cols;
    Teuchos::ArrayView<const trilinos::Scalar> vals;
    A->getLocalRowView(static_cast<trilinos::LO>(lrow), cols, vals);
    for (trilinos::LO i = 0; i < cols.size(); ++i) {
        if (cols[i] == static_cast<trilinos::LO>(lcol)) {
            return static_cast<Real>(vals[i]);
        }
    }
    return 0.0;
}

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_TRILINOS
