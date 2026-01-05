/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_TRILINOS_VECTOR_H
#define SVMP_FE_BACKENDS_TRILINOS_VECTOR_H

#include "Backends/Interfaces/GenericVector.h"

#if defined(FE_HAS_TRILINOS)

#include "Backends/Trilinos/TrilinosUtils.h"

#include <Teuchos_RCP.hpp>

#include <vector>

namespace svmp {
namespace FE {
namespace backends {

class TrilinosVector final : public GenericVector {
public:
    explicit TrilinosVector(GlobalIndex global_size);
    TrilinosVector(GlobalIndex local_size, GlobalIndex global_size);
    TrilinosVector(GlobalIndex owned_first,
                   GlobalIndex local_owned_size,
                   GlobalIndex global_size,
                   const std::vector<GlobalIndex>& ghost_global_indices);

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::Trilinos; }
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

    [[nodiscard]] Teuchos::RCP<trilinos::Vector> tpetra() const
    {
        syncVectorFromCache();
        return vec_;
    }
    [[nodiscard]] Teuchos::RCP<const trilinos::Map> map() const { return map_; }

    void invalidateLocalCache() const noexcept;

private:
    void syncCacheFromVector() const;
    void syncVectorFromCache() const;

    Teuchos::RCP<const trilinos::Map> map_{};
    Teuchos::RCP<trilinos::Vector> vec_{};
    Teuchos::RCP<const trilinos::Map> overlap_map_{};
    Teuchos::RCP<trilinos::Vector> overlap_vec_{};
    Teuchos::RCP<const Tpetra::Import<trilinos::LO, trilinos::GO, trilinos::Node>> importer_{};

    mutable std::vector<Real> local_cache_{};
    mutable bool local_cache_valid_{false};
    mutable bool local_cache_dirty_{false};
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_TRILINOS

#endif // SVMP_FE_BACKENDS_TRILINOS_VECTOR_H
