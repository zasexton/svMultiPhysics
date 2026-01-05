/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_PETSC_MATRIX_H
#define SVMP_FE_BACKENDS_PETSC_MATRIX_H

#include "Backends/Interfaces/GenericMatrix.h"

#if defined(FE_HAS_PETSC)

#include "Backends/PETSc/PetscUtils.h"

#include <petscmat.h>

namespace svmp {
namespace FE {
namespace sparsity {
class DistributedSparsityPattern;
} // namespace sparsity

namespace backends {

class PetscMatrix final : public GenericMatrix {
public:
    explicit PetscMatrix(const sparsity::SparsityPattern& sparsity);
    explicit PetscMatrix(const sparsity::DistributedSparsityPattern& sparsity);
    ~PetscMatrix() override;

    PetscMatrix(PetscMatrix&& other) noexcept;
    PetscMatrix& operator=(PetscMatrix&& other) noexcept;

    PetscMatrix(const PetscMatrix&) = delete;
    PetscMatrix& operator=(const PetscMatrix&) = delete;

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::PETSc; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override;
    [[nodiscard]] GlobalIndex numCols() const noexcept override;

    void zero() override;
    void finalizeAssembly() override;

    void mult(const GenericVector& x, GenericVector& y) const override;
    void multAdd(const GenericVector& x, GenericVector& y) const override;

    [[nodiscard]] std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() override;

    [[nodiscard]] Real getEntry(GlobalIndex row, GlobalIndex col) const override;

    [[nodiscard]] Mat petsc() const noexcept { return mat_; }

private:
    void allocateFromSequential(const sparsity::SparsityPattern& sparsity);
    void allocateFromDistributed(const sparsity::DistributedSparsityPattern& sparsity);

    Mat mat_{nullptr};
    mutable Vec work_{nullptr};
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_PETSC

#endif // SVMP_FE_BACKENDS_PETSC_MATRIX_H
