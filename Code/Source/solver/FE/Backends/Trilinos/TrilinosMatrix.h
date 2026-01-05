/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_TRILINOS_MATRIX_H
#define SVMP_FE_BACKENDS_TRILINOS_MATRIX_H

#include "Backends/Interfaces/GenericMatrix.h"

#if defined(FE_HAS_TRILINOS)

#include "Backends/Trilinos/TrilinosUtils.h"

#include <Teuchos_RCP.hpp>

namespace svmp {
namespace FE {
namespace sparsity {
class DistributedSparsityPattern;
} // namespace sparsity

namespace backends {

class TrilinosMatrix final : public GenericMatrix {
public:
    explicit TrilinosMatrix(const sparsity::SparsityPattern& sparsity);
    explicit TrilinosMatrix(const sparsity::DistributedSparsityPattern& sparsity);

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::Trilinos; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override;
    [[nodiscard]] GlobalIndex numCols() const noexcept override;

    void zero() override;
    void finalizeAssembly() override;
    void mult(const GenericVector& x, GenericVector& y) const override;
    void multAdd(const GenericVector& x, GenericVector& y) const override;

    [[nodiscard]] std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() override;
    [[nodiscard]] Real getEntry(GlobalIndex row, GlobalIndex col) const override;

    [[nodiscard]] Teuchos::RCP<trilinos::CrsMatrix> tpetra() const { return mat_; }

private:
    void allocateFromSequential(const sparsity::SparsityPattern& sparsity);
    void allocateFromDistributed(const sparsity::DistributedSparsityPattern& sparsity);

    Teuchos::RCP<trilinos::CrsMatrix> mat_{};
    Teuchos::RCP<const trilinos::Map> row_map_{};
    Teuchos::RCP<const trilinos::Map> domain_map_{};
    Teuchos::RCP<const trilinos::Map> range_map_{};
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_TRILINOS

#endif // SVMP_FE_BACKENDS_TRILINOS_MATRIX_H
