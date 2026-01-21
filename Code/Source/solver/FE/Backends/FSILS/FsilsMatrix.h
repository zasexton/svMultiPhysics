/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_FSILS_MATRIX_H
#define SVMP_FE_BACKENDS_FSILS_MATRIX_H

#include "Backends/Interfaces/GenericMatrix.h"
#include "Backends/Interfaces/DofPermutation.h"
#include "Backends/FSILS/FsilsShared.h"

#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace sparsity {
class SparsityPattern;
class DistributedSparsityPattern;
} // namespace sparsity

namespace backends {

class FsilsMatrix final : public GenericMatrix {
public:
    explicit FsilsMatrix(const sparsity::SparsityPattern& sparsity);
    FsilsMatrix(const sparsity::SparsityPattern& sparsity,
                int dof_per_node,
                std::shared_ptr<const DofPermutation> dof_permutation = {});
    FsilsMatrix(const sparsity::DistributedSparsityPattern& sparsity,
                int dof_per_node,
                std::shared_ptr<const DofPermutation> dof_permutation = {});
    ~FsilsMatrix() override;

    FsilsMatrix(FsilsMatrix&&) noexcept;
    FsilsMatrix& operator=(FsilsMatrix&&) noexcept;

    FsilsMatrix(const FsilsMatrix&) = delete;
    FsilsMatrix& operator=(const FsilsMatrix&) = delete;

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::FSILS; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override;
    [[nodiscard]] GlobalIndex numCols() const noexcept override;

    void zero() override;
    void finalizeAssembly() override;
    void mult(const GenericVector& x, GenericVector& y) const override;
    void multAdd(const GenericVector& x, GenericVector& y) const override;

    [[nodiscard]] std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() override;
    [[nodiscard]] Real getEntry(GlobalIndex row, GlobalIndex col) const override;

    void addValue(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode);

    [[nodiscard]] std::shared_ptr<const FsilsShared> shared() const noexcept { return shared_; }

    // Internal access for solver integration
    [[nodiscard]] int fsilsDof() const noexcept;
    [[nodiscard]] void* fsilsLhsPtr() noexcept;
    [[nodiscard]] const void* fsilsLhsPtr() const noexcept;
    [[nodiscard]] Real* fsilsValuesPtr() noexcept;
    [[nodiscard]] const Real* fsilsValuesPtr() const noexcept;
    [[nodiscard]] GlobalIndex fsilsNnz() const noexcept;

private:
    GlobalIndex global_rows_{0};
    GlobalIndex global_cols_{0};
    GlobalIndex nnz_{0};

    std::shared_ptr<FsilsShared> shared_{};
    std::vector<Real> values_{}; // (dof*dof) x nnz (column-major for FSILS Array wrapper)
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_FSILS_MATRIX_H
