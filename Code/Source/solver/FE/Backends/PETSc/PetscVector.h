/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_PETSC_VECTOR_H
#define SVMP_FE_BACKENDS_PETSC_VECTOR_H

#include "Backends/Interfaces/GenericVector.h"

#if defined(FE_HAS_PETSC)

#include "Backends/PETSc/PetscUtils.h"

#include <petscvec.h>

#include <vector>

namespace svmp {
namespace FE {
namespace backends {

class PetscVector final : public GenericVector {
public:
    explicit PetscVector(GlobalIndex global_size);
    PetscVector(GlobalIndex local_size, GlobalIndex global_size);
    PetscVector(GlobalIndex local_size, GlobalIndex global_size, const std::vector<GlobalIndex>& ghost_global_indices);
    ~PetscVector() override;

    PetscVector(PetscVector&& other) noexcept;
    PetscVector& operator=(PetscVector&& other) noexcept;

    PetscVector(const PetscVector&) = delete;
    PetscVector& operator=(const PetscVector&) = delete;

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::PETSc; }
    [[nodiscard]] GlobalIndex size() const noexcept override;

    void zero() override;
    void set(Real value) override;
    void add(Real value) override;
    void scale(Real alpha) override;

    [[nodiscard]] Real dot(const GenericVector& other) const override;
    [[nodiscard]] Real norm() const override;

    void updateGhosts() override;

    [[nodiscard]] std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() override;

    [[nodiscard]] std::span<Real> localSpan() override;
    [[nodiscard]] std::span<const Real> localSpan() const override;

    [[nodiscard]] Vec petsc() const;

    void invalidateLocalCache() const noexcept;

private:
    void ensureVecUpToDate() const;
    void ensureCacheUpToDate() const;

    PetscInt local_owned_{0};
    PetscInt ghost_count_{0};
    bool ghosted_{false};

    mutable Vec vec_{nullptr};
    mutable std::vector<Real> local_cache_{};
    mutable bool local_cache_valid_{false};
    mutable bool local_cache_dirty_{false};
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_PETSC

#endif // SVMP_FE_BACKENDS_PETSC_VECTOR_H
