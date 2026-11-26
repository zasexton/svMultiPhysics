/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_EIGEN_MATRIX_H
#define SVMP_FE_BACKENDS_EIGEN_MATRIX_H

#include "Backends/Interfaces/GenericMatrix.h"

#if defined(FE_HAS_EIGEN)
#include "Backends/Interfaces/MatrixAssemblyError.h"
#include <Eigen/Sparse>
#endif

namespace svmp {
namespace FE {
namespace sparsity {
class SparsityPattern;
} // namespace sparsity

namespace backends {

#if defined(FE_HAS_EIGEN)

class EigenMatrix final : public GenericMatrix {
public:
    using StorageIndex = int;
    using SparseMat = Eigen::SparseMatrix<Real, Eigen::RowMajor, StorageIndex>;

    explicit EigenMatrix(const sparsity::SparsityPattern& sparsity);

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::Eigen; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return n_rows_; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return n_cols_; }

    void zero() override;
    void finalizeAssembly() override;
    void mult(const GenericVector& x, GenericVector& y) const override;
    void multAdd(const GenericVector& x, GenericVector& y) const override;

    [[nodiscard]] std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() override;

    [[nodiscard]] Real getEntry(GlobalIndex row, GlobalIndex col) const override;

    bool reinitFromPattern(const sparsity::SparsityPattern& pattern) override;

    [[nodiscard]] SparseMat& eigen() noexcept { return mat_; }
    [[nodiscard]] const SparseMat& eigen() const noexcept { return mat_; }

    void addValue(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode);

private:
    friend class EigenMatrixView;

    bool resolveAssemblySlot(GlobalIndex row, GlobalIndex col, Real value, StorageIndex& slot);
    void writeAssemblySlot(StorageIndex slot, Real value, assembly::AddMode mode);
    void initializeFromPattern(const sparsity::SparsityPattern& pattern);

    GlobalIndex n_rows_{0};
    GlobalIndex n_cols_{0};
    SparseMat mat_;
    MatrixAssemblyError assembly_error_;
};

#endif // FE_HAS_EIGEN

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_EIGEN_MATRIX_H
