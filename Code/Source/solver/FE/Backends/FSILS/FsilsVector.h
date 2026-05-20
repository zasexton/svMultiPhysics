/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_FSILS_VECTOR_H
#define SVMP_FE_BACKENDS_FSILS_VECTOR_H

#include "Backends/Interfaces/GenericVector.h"
#include "Backends/FSILS/FsilsShared.h"

#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

class FsilsVector final : public GenericVector {
public:
    explicit FsilsVector(GlobalIndex global_size);
    explicit FsilsVector(std::shared_ptr<const FsilsShared> shared);

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::FSILS; }
    [[nodiscard]] GlobalIndex size() const noexcept override { return global_size_; }
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

    /**
     * @brief Accumulate a raw contribution buffer and refresh ghost entries.
     *
     * Use this only for buffers that intentionally contain independently
     * assembled contributions on both owned and ghost entries. Ordinary solver
     * vectors should use updateGhosts() instead.
     */
    void accumulateRawContributionsAndUpdateGhosts();

    [[nodiscard]] std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() override;

    [[nodiscard]] std::span<Real> localSpan() override;
    [[nodiscard]] std::span<const Real> localSpan() const override;

    [[nodiscard]] const FsilsShared* shared() const noexcept { return shared_.get(); }
    [[nodiscard]] bool usesOwnedRowLayout() const noexcept;
    [[nodiscard]] bool ownsFeDof(GlobalIndex fe_dof) const noexcept;
    [[nodiscard]] std::vector<GlobalIndex> ownedFeDofs() const;

    [[nodiscard]] std::vector<Real>& data() noexcept
    {
        markModified();
        return data_;
    }
    [[nodiscard]] const std::vector<Real>& data() const noexcept { return data_; }

    // Resolve FE DOFs into backend-local vector entries and reuse the mapping
    // across repeated element/face gathers with the same connectivity.
    void resolveEntriesCached(std::span<const GlobalIndex> dofs,
                              std::span<GlobalIndex> resolved) const;

private:
    struct ResolutionCacheEntry {
        std::vector<GlobalIndex> dofs{};
        std::vector<GlobalIndex> resolved{};
    };

    void exchangeOwnedHalo(bool owner_to_ghost_only);

    GlobalIndex global_size_{0};
    std::shared_ptr<const FsilsShared> shared_{};
    std::vector<Real> data_;
    mutable std::vector<double> halo_internal_work_{};
    mutable std::unordered_map<std::size_t, std::vector<ResolutionCacheEntry>> resolution_cache_{};
    std::uint64_t value_revision_{0};
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_FSILS_VECTOR_H
