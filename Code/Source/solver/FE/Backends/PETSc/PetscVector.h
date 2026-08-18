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

#include <unordered_map>
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
    [[nodiscard]] std::uint64_t valueRevision() const noexcept override { return value_revision_; }
    void markModified() noexcept override { ++value_revision_; }

    void zero() override;
    void set(Real value) override;
    void add(Real value) override;
    void scale(Real alpha) override;

    void copyFrom(const GenericVector& other) override;

    [[nodiscard]] Real dot(const GenericVector& other) const override;
    [[nodiscard]] Real norm() const override;

    void updateGhosts() override;
    [[nodiscard]] bool ghostUpdateRequiresCollectiveParticipation()
        const noexcept override
    {
        return ghosted_;
    }

    [[nodiscard]] std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() override;
    [[nodiscard]] std::unique_ptr<assembly::GlobalSystemView>
    createGhostedReadView() override;

    [[nodiscard]] std::span<Real> localSpan() override;
    [[nodiscard]] std::span<const Real> localSpan() const override;
    [[nodiscard]] std::vector<GlobalIndex> ownedGlobalRows() const override;

    [[nodiscard]] Vec petsc() const;

    /**
     * @brief Read owned or refreshed ghost entries by public global index.
     *
     * Call updateGhosts() collectively before reading overlap entries. A
     * valid global index outside this vector's owned/ghost layout is rejected
     * rather than being sampled as zero.
     */
    void readGlobalEntries(std::span<const GlobalIndex> dofs,
                           std::span<Real> values) const;

    void invalidateLocalCache() const noexcept;

private:
    void ensureVecUpToDate() const;
    void ensureCacheUpToDate() const;

    PetscInt local_owned_{0};
    PetscInt ghost_count_{0};
    bool ghosted_{false};
    std::unordered_map<GlobalIndex, PetscInt> ghost_local_indices_{};

    mutable Vec vec_{nullptr};
    mutable std::vector<Real> local_cache_{};
    mutable bool local_cache_valid_{false};
    mutable bool local_cache_dirty_{false};
    std::uint64_t value_revision_{0};
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_PETSC

#endif // SVMP_FE_BACKENDS_PETSC_VECTOR_H
