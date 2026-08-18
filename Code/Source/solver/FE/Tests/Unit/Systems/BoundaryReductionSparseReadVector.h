/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TESTS_BOUNDARY_REDUCTION_SPARSE_READ_VECTOR_H
#define SVMP_FE_TESTS_BOUNDARY_REDUCTION_SPARSE_READ_VECTOR_H

#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp::FE::systems::testing {

/**
 * Test vector with O(locally-relevant rows) storage and an independently
 * reported global size. ownedGlobalRows() deliberately rejects so a test fails
 * if a reduction regresses to owner enumeration or a global dense snapshot.
 */
class BoundaryReductionSparseReadVector final
    : public backends::GenericVector {
public:
    BoundaryReductionSparseReadVector(
        GlobalIndex global_size,
        std::unordered_map<GlobalIndex, Real> locally_relevant_values)
        : global_size_(global_size),
          values_(std::move(locally_relevant_values))
    {
        FE_THROW_IF(global_size_ <= 0,
                    InvalidArgumentException,
                    "BoundaryReductionSparseReadVector: invalid global size");
        for (const auto& [dof, value] : values_) {
            FE_THROW_IF(dof < 0 || dof >= global_size_ || !std::isfinite(value),
                        InvalidArgumentException,
                        "BoundaryReductionSparseReadVector: invalid local entry");
        }
    }

    [[nodiscard]] backends::BackendKind backendKind() const noexcept override
    {
        return backends::BackendKind::Eigen;
    }

    [[nodiscard]] GlobalIndex size() const noexcept override
    {
        return global_size_;
    }

    [[nodiscard]] std::uint64_t valueRevision() const noexcept override
    {
        return revision_;
    }

    void markModified() noexcept override { ++revision_; }

    void zero() override
    {
        for (auto& [dof, value] : values_) {
            (void)dof;
            value = Real{0.0};
        }
        markModified();
    }

    void set(Real value) override
    {
        for (auto& [dof, entry] : values_) {
            (void)dof;
            entry = value;
        }
        markModified();
    }

    void add(Real value) override
    {
        for (auto& [dof, entry] : values_) {
            (void)dof;
            entry += value;
        }
        markModified();
    }

    void scale(Real alpha) override
    {
        for (auto& [dof, entry] : values_) {
            (void)dof;
            entry *= alpha;
        }
        markModified();
    }

    void copyFrom(const backends::GenericVector&) override
    {
        FE_THROW(NotImplementedException,
                 "BoundaryReductionSparseReadVector::copyFrom is not used");
    }

    [[nodiscard]] Real dot(const backends::GenericVector&) const override
    {
        FE_THROW(NotImplementedException,
                 "BoundaryReductionSparseReadVector::dot is not used");
    }

    [[nodiscard]] Real norm() const override
    {
        Real sum = 0.0;
        for (const auto& [dof, value] : values_) {
            (void)dof;
            sum += value * value;
        }
        return std::sqrt(sum);
    }

    void updateGhosts() override { ++ghost_refresh_count_; }
    [[nodiscard]] bool ghostUpdateRequiresCollectiveParticipation()
        const noexcept override
    {
        return collective_ghost_refresh_required_;
    }
    void setCollectiveGhostRefreshRequirement(bool required) noexcept
    {
        collective_ghost_refresh_required_ = required;
    }

    [[nodiscard]] std::unique_ptr<assembly::GlobalSystemView>
    createAssemblyView() override;

    [[nodiscard]] std::span<Real> localSpan() override { return {}; }
    [[nodiscard]] std::span<const Real> localSpan() const override { return {}; }

    [[nodiscard]] std::vector<GlobalIndex> ownedGlobalRows() const override
    {
        FE_THROW(FEException,
                 "BoundaryReductionSparseReadVector: owner enumeration is forbidden");
    }

    [[nodiscard]] Real read(GlobalIndex dof) const
    {
        const auto entry = values_.find(dof);
        FE_THROW_IF(entry == values_.end(),
                    FEException,
                    "BoundaryReductionSparseReadVector: requested row " +
                        std::to_string(dof) +
                        " is outside the local read overlap");
        ++read_count_;
        return entry->second;
    }

    [[nodiscard]] std::size_t locallyRelevantCount() const noexcept
    {
        return values_.size();
    }

    [[nodiscard]] std::uint64_t readCount() const noexcept { return read_count_; }
    [[nodiscard]] std::uint64_t ghostRefreshCount() const noexcept
    {
        return ghost_refresh_count_;
    }

private:
    class View final : public assembly::GlobalSystemView {
    public:
        explicit View(BoundaryReductionSparseReadVector& vector)
            : vector_(&vector)
        {
        }

        void addMatrixEntries(std::span<const GlobalIndex>,
                              std::span<const Real>,
                              assembly::AddMode) override {}
        void addMatrixEntries(std::span<const GlobalIndex>,
                              std::span<const GlobalIndex>,
                              std::span<const Real>,
                              assembly::AddMode) override {}
        void addMatrixEntry(GlobalIndex, GlobalIndex, Real,
                            assembly::AddMode) override {}
        void setDiagonal(std::span<const GlobalIndex>,
                         std::span<const Real>) override {}
        void setDiagonal(GlobalIndex, Real) override {}
        void zeroRows(std::span<const GlobalIndex>, bool) override {}

        void addVectorEntries(std::span<const GlobalIndex>,
                              std::span<const Real>,
                              assembly::AddMode) override
        {
            FE_THROW(NotImplementedException,
                     "BoundaryReductionSparseReadVector::View is read-only");
        }
        void addVectorEntry(GlobalIndex, Real, assembly::AddMode) override
        {
            FE_THROW(NotImplementedException,
                     "BoundaryReductionSparseReadVector::View is read-only");
        }
        void setVectorEntries(std::span<const GlobalIndex>,
                              std::span<const Real>) override
        {
            FE_THROW(NotImplementedException,
                     "BoundaryReductionSparseReadVector::View is read-only");
        }
        void zeroVectorEntries(std::span<const GlobalIndex>) override
        {
            FE_THROW(NotImplementedException,
                     "BoundaryReductionSparseReadVector::View is read-only");
        }

        [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override
        {
            FE_CHECK_NOT_NULL(vector_,
                              "BoundaryReductionSparseReadVector::View::vector");
            return vector_->read(dof);
        }

        void getVectorEntries(std::span<const GlobalIndex> dofs,
                              std::span<Real> out) const override
        {
            FE_THROW_IF(dofs.size() != out.size(),
                        InvalidArgumentException,
                        "BoundaryReductionSparseReadVector::View: size mismatch");
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                out[i] = getVectorEntry(dofs[i]);
            }
        }

        void beginAssemblyPhase() override
        {
            phase_ = assembly::AssemblyPhase::Building;
        }
        void endAssemblyPhase() override
        {
            phase_ = assembly::AssemblyPhase::Flushing;
        }
        void finalizeAssembly() override
        {
            phase_ = assembly::AssemblyPhase::Finalized;
        }
        [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override
        {
            return phase_;
        }

        [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
        [[nodiscard]] bool hasVector() const noexcept override { return true; }
        [[nodiscard]] GlobalIndex numRows() const noexcept override
        {
            return vector_ != nullptr ? vector_->size() : 0;
        }
        [[nodiscard]] GlobalIndex numCols() const noexcept override { return 1; }
        [[nodiscard]] bool isDistributed() const noexcept override { return true; }
        [[nodiscard]] std::string backendName() const override
        {
            return "BoundaryReductionSparseReadVector";
        }
        void zero() override
        {
            FE_THROW(NotImplementedException,
                     "BoundaryReductionSparseReadVector::View is read-only");
        }

    private:
        BoundaryReductionSparseReadVector* vector_{nullptr};
        assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
    };

    GlobalIndex global_size_{0};
    std::unordered_map<GlobalIndex, Real> values_{};
    std::uint64_t revision_{0};
    mutable std::uint64_t read_count_{0};
    std::uint64_t ghost_refresh_count_{0};
    bool collective_ghost_refresh_required_{false};
};

inline std::unique_ptr<assembly::GlobalSystemView>
BoundaryReductionSparseReadVector::createAssemblyView()
{
    return std::make_unique<View>(*this);
}

} // namespace svmp::FE::systems::testing

#endif // SVMP_FE_TESTS_BOUNDARY_REDUCTION_SPARSE_READ_VECTOR_H
