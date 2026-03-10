/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/FSILS/FsilsVector.h"

#include "Backends/FSILS/liner_solver/fsils_api.hpp"
#include "Backends/FSILS/liner_solver/fils_struct.hpp"
#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <mpi.h>
#include <string>

namespace svmp {
namespace FE {
namespace backends {

namespace {

[[nodiscard]] std::size_t hashGlobalIndexSpan(std::span<const GlobalIndex> values) noexcept
{
    std::uint64_t hash = 1469598103934665603ull;
    for (const auto value : values) {
        const auto mixed = static_cast<std::uint64_t>(std::hash<GlobalIndex>{}(value));
        hash ^= mixed + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    }
    hash ^= static_cast<std::uint64_t>(values.size()) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    return static_cast<std::size_t>(hash);
}

[[nodiscard]] bool spanMatches(std::span<const GlobalIndex> lhs,
                               const std::vector<GlobalIndex>& rhs) noexcept
{
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

void resolveFsilsVectorEntriesUncached(const FsilsVector& vec,
                                       std::span<const GlobalIndex> dofs,
                                       std::span<GlobalIndex> resolved)
{
    FE_THROW_IF(dofs.size() != resolved.size(), InvalidArgumentException,
                "resolveFsilsVectorEntriesUncached: size mismatch");

    const auto* shared = vec.shared();
    const GlobalIndex vec_size = vec.size();
    if (!shared) {
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            const auto dof = dofs[i];
            resolved[i] = (dof >= 0 && dof < vec_size) ? dof : INVALID_GLOBAL_INDEX;
        }
        return;
    }

    const auto* perm_ptr = shared->dof_permutation.get();
    const bool has_perm = (perm_ptr != nullptr && !perm_ptr->forward.empty());
    const auto* fwd_data = has_perm ? perm_ptr->forward.data() : nullptr;
    const auto fwd_size = has_perm ? perm_ptr->forward.size() : std::size_t{0};
    const int dof_per_node = shared->dof;

    int cached_global_node = -1;
    int cached_old_node = -1;

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        auto dof = dofs[i];
        if (dof < 0 || dof >= vec_size) {
            resolved[i] = INVALID_GLOBAL_INDEX;
            continue;
        }

        if (has_perm) {
            if (static_cast<std::size_t>(dof) >= fwd_size) {
                resolved[i] = INVALID_GLOBAL_INDEX;
                continue;
            }
            dof = fwd_data[static_cast<std::size_t>(dof)];
            if (dof < 0 || dof >= vec_size) {
                resolved[i] = INVALID_GLOBAL_INDEX;
                continue;
            }
        }

        const int global_node = static_cast<int>(dof / dof_per_node);
        const int comp = static_cast<int>(dof % dof_per_node);

        int old_node = cached_old_node;
        if (global_node != cached_global_node) {
            old_node = shared->globalNodeToOld(global_node);
            cached_global_node = global_node;
            cached_old_node = old_node;
        }

        if (old_node < 0) {
            resolved[i] = INVALID_GLOBAL_INDEX;
            continue;
        }

        resolved[i] = static_cast<GlobalIndex>(
            static_cast<std::size_t>(old_node) * static_cast<std::size_t>(dof_per_node) +
            static_cast<std::size_t>(comp));
    }
}

void gatherFsilsVectorEntries(const FsilsVector& vec,
                              std::span<const GlobalIndex> resolved,
                              std::span<Real> out)
{
    FE_THROW_IF(resolved.size() != out.size(), InvalidArgumentException,
                "gatherFsilsVectorEntries: size mismatch");

    const auto& data = vec.data();
    const auto data_size = static_cast<GlobalIndex>(data.size());
    for (std::size_t i = 0; i < resolved.size(); ++i) {
        const auto idx = resolved[i];
        out[i] = (idx >= 0 && idx < data_size) ? data[static_cast<std::size_t>(idx)] : 0.0;
    }
}

class FsilsVectorView final : public assembly::GlobalSystemView {
public:
    explicit FsilsVectorView(FsilsVector& vec) : vec_(&vec) {}

    void addMatrixEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
    void addMatrixEntries(std::span<const GlobalIndex>, std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
    void addMatrixEntry(GlobalIndex, GlobalIndex, Real, assembly::AddMode) override {}
    void setDiagonal(std::span<const GlobalIndex>, std::span<const Real>) override {}
    void setDiagonal(GlobalIndex, Real) override {}
    void zeroRows(std::span<const GlobalIndex>, bool) override {}

    void addVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_vector,
                          assembly::AddMode mode) override
    {
        if (dofs.size() != local_vector.size()) {
            FE_THROW(InvalidArgumentException, "FsilsVectorView::addVectorEntries: size mismatch");
        }
        FE_CHECK_NOT_NULL(vec_, "FsilsVectorView::vec");

        thread_local std::vector<GlobalIndex> resolved;
        resolved.resize(dofs.size());
        vec_->resolveEntriesCached(dofs, resolved);

        auto& data = vec_->data();
        const auto data_size = static_cast<GlobalIndex>(data.size());
        for (std::size_t i = 0; i < resolved.size(); ++i) {
            const auto idx = resolved[i];
            if (idx < 0 || idx >= data_size) {
                continue;
            }

            auto& dst = data[static_cast<std::size_t>(idx)];
            switch (mode) {
                case assembly::AddMode::Add:
                    dst += local_vector[i];
                    break;
                case assembly::AddMode::Insert:
                    dst = local_vector[i];
                    break;
                case assembly::AddMode::Max:
                    dst = std::max(dst, local_vector[i]);
                    break;
                case assembly::AddMode::Min:
                    dst = std::min(dst, local_vector[i]);
                    break;
            }
        }
    }

    void addVectorEntriesResolved(std::span<const GlobalIndex> dofs,
                                   std::span<const GlobalIndex> resolved,
                                   std::span<const Real> local_vector,
                                   assembly::AddMode mode) override
    {
        (void)dofs;
        FE_CHECK_NOT_NULL(vec_, "FsilsVectorView::vec");

        auto& data = vec_->data();
        const auto data_size = static_cast<GlobalIndex>(data.size());
        for (std::size_t i = 0; i < resolved.size(); ++i) {
            const auto idx = resolved[i];
            if (idx < 0 || idx >= data_size) {
                continue;
            }

            auto& dst = data[static_cast<std::size_t>(idx)];
            switch (mode) {
                case assembly::AddMode::Add:
                    dst += local_vector[i];
                    break;
                case assembly::AddMode::Insert:
                    dst = local_vector[i];
                    break;
                case assembly::AddMode::Max:
                    dst = std::max(dst, local_vector[i]);
                    break;
                case assembly::AddMode::Min:
                    dst = std::min(dst, local_vector[i]);
                    break;
            }
        }
    }

    void addVectorEntry(GlobalIndex dof, Real value, assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(vec_, "FsilsVectorView::vec");
        GlobalIndex resolved = INVALID_GLOBAL_INDEX;
        vec_->resolveEntriesCached(std::span<const GlobalIndex>(&dof, 1),
                                   std::span<GlobalIndex>(&resolved, 1));
        if (resolved < 0 || static_cast<std::size_t>(resolved) >= vec_->data().size()) {
            return;
        }

        auto& dst = vec_->data()[static_cast<std::size_t>(resolved)];
        switch (mode) {
            case assembly::AddMode::Add:
                dst += value;
                break;
            case assembly::AddMode::Insert:
                dst = value;
                break;
            case assembly::AddMode::Max:
                dst = std::max(dst, value);
                break;
            case assembly::AddMode::Min:
                dst = std::min(dst, value);
                break;
        }
    }

    void setVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> values) override
    {
        addVectorEntries(dofs, values, assembly::AddMode::Insert);
    }

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override
    {
        thread_local std::vector<Real> zeros;
        zeros.assign(dofs.size(), 0.0);
        addVectorEntries(dofs, zeros, assembly::AddMode::Insert);
    }

    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override
    {
        FE_CHECK_NOT_NULL(vec_, "FsilsVectorView::vec");
        GlobalIndex resolved = INVALID_GLOBAL_INDEX;
        vec_->resolveEntriesCached(std::span<const GlobalIndex>(&dof, 1),
                                   std::span<GlobalIndex>(&resolved, 1));
        if (resolved < 0 || static_cast<std::size_t>(resolved) >= vec_->data().size()) {
            return 0.0;
        }
        return vec_->data()[static_cast<std::size_t>(resolved)];
    }

    [[nodiscard]] const void* vectorLayoutHandle() const noexcept override
    {
        if (vec_ == nullptr) {
            return nullptr;
        }
        if (const auto* shared = vec_->shared()) {
            return shared;
        }
        return vec_;
    }

    void resolveVectorEntries(std::span<const GlobalIndex> dofs,
                              std::span<GlobalIndex> resolved) const override
    {
        FE_CHECK_NOT_NULL(vec_, "FsilsVectorView::vec");
        vec_->resolveEntriesCached(dofs, resolved);
    }

    void getVectorEntriesResolved(std::span<const GlobalIndex> resolved,
                                  std::span<Real> out) const override
    {
        FE_CHECK_NOT_NULL(vec_, "FsilsVectorView::vec");
        gatherFsilsVectorEntries(*vec_, resolved, out);
    }

    void getVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<Real> out) const override
    {
        FE_CHECK_NOT_NULL(vec_, "FsilsVectorView::vec");
        thread_local std::vector<GlobalIndex> resolved;
        resolved.resize(dofs.size());
        vec_->resolveEntriesCached(dofs, resolved);
        gatherFsilsVectorEntries(*vec_, resolved, out);
    }

    void beginAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Building; }
    void endAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Flushing; }
    void finalizeAssembly() override { phase_ = assembly::AssemblyPhase::Finalized; }
    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return phase_; }

    [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
    [[nodiscard]] bool hasVector() const noexcept override { return true; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return vec_ ? vec_->size() : 0; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return 1; }
    [[nodiscard]] std::string backendName() const override { return "FSILSVector"; }

    void zero() override
    {
        FE_CHECK_NOT_NULL(vec_, "FsilsVectorView::vec");
        vec_->zero();
    }

private:
    FsilsVector* vec_{nullptr};
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
};

} // namespace

FsilsVector::FsilsVector(GlobalIndex size)
{
    FE_THROW_IF(size < 0, InvalidArgumentException, "FsilsVector: negative size");
    global_size_ = size;
    data_.assign(static_cast<std::size_t>(size), 0.0);
}

FsilsVector::FsilsVector(std::shared_ptr<const FsilsShared> shared)
{
    FE_CHECK_NOT_NULL(shared.get(), "FsilsVector: shared layout");
    FE_THROW_IF(shared->global_dofs < 0, InvalidArgumentException, "FsilsVector: negative global size");
    FE_THROW_IF(shared->dof <= 0, InvalidArgumentException, "FsilsVector: invalid dof");
    FE_THROW_IF(shared->lhs.nNo < 0, InvalidArgumentException, "FsilsVector: invalid local node count");

    global_size_ = shared->global_dofs;
    shared_ = std::move(shared);
    const std::size_t local_size =
        static_cast<std::size_t>(shared_->dof) * static_cast<std::size_t>(shared_->lhs.nNo);
    data_.assign(local_size, 0.0);
}

void FsilsVector::resolveEntriesCached(std::span<const GlobalIndex> dofs,
                                       std::span<GlobalIndex> resolved) const
{
    FE_THROW_IF(dofs.size() != resolved.size(), InvalidArgumentException,
                "FsilsVector::resolveEntriesCached: size mismatch");

    const auto hash = hashGlobalIndexSpan(dofs);
    auto bucket_it = resolution_cache_.find(hash);
    if (bucket_it != resolution_cache_.end()) {
        for (const auto& entry : bucket_it->second) {
            if (spanMatches(dofs, entry.dofs)) {
                std::copy(entry.resolved.begin(), entry.resolved.end(), resolved.begin());
                return;
            }
        }
    }

    auto& bucket = resolution_cache_[hash];
    auto& entry = bucket.emplace_back();
    entry.dofs.assign(dofs.begin(), dofs.end());
    entry.resolved.resize(dofs.size());
    resolveFsilsVectorEntriesUncached(*this, dofs, entry.resolved);
    std::copy(entry.resolved.begin(), entry.resolved.end(), resolved.begin());
}

void FsilsVector::zero()
{
    std::fill(data_.begin(), data_.end(), 0.0);
}

void FsilsVector::set(Real value)
{
    std::fill(data_.begin(), data_.end(), value);
}

void FsilsVector::add(Real value)
{
    for (auto& v : data_) {
        v += value;
    }
}

void FsilsVector::scale(Real alpha)
{
    for (auto& v : data_) {
        v *= alpha;
    }
}

void FsilsVector::copyFrom(const GenericVector& other)
{
    const auto* o = dynamic_cast<const FsilsVector*>(&other);
    FE_THROW_IF(!o, InvalidArgumentException, "FsilsVector::copyFrom: backend mismatch");
    FE_THROW_IF(size() != o->size(), InvalidArgumentException, "FsilsVector::copyFrom: size mismatch");
    std::copy(o->data_.begin(), o->data_.end(), data_.begin());
}

Real FsilsVector::dot(const GenericVector& other) const
{
    const auto* o = dynamic_cast<const FsilsVector*>(&other);
    FE_THROW_IF(!o, InvalidArgumentException, "FsilsVector::dot: backend mismatch");
    FE_THROW_IF(o->global_size_ != global_size_, InvalidArgumentException, "FsilsVector::dot: global size mismatch");
    FE_THROW_IF(o->data_.size() != data_.size(), InvalidArgumentException, "FsilsVector::dot: local size mismatch");
    FE_THROW_IF(o->shared_ != shared_, InvalidArgumentException, "FsilsVector::dot: layout mismatch");

    Real sum = 0.0;

    if (shared_) {
        const int dof = shared_->dof;
        const int nNo = shared_->lhs.nNo;
        const int mynNo = shared_->lhs.mynNo;
        const auto& lhs = shared_->lhs;
        for (int old = 0; old < nNo; ++old) {
            if (lhs.map(old) >= mynNo) continue;
            const std::size_t base = static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
            for (int c = 0; c < dof; ++c) {
                sum += data_[base + static_cast<std::size_t>(c)] * o->data_[base + static_cast<std::size_t>(c)];
            }
        }

        if (lhs.commu.nTasks != 1) {
            Real global_sum = 0.0;
            MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, lhs.commu.comm);
            return global_sum;
        }
        return sum;
    }

    for (std::size_t i = 0; i < data_.size(); ++i) {
        sum += data_[i] * o->data_[i];
    }
    return sum;
}

Real FsilsVector::norm() const
{
    return std::sqrt(dot(*this));
}

void FsilsVector::updateGhosts()
{
    if (!shared_) {
        return;
    }

    const auto& lhs = shared_->lhs;
    if (lhs.commu.nTasks == 1) {
        return;
    }

    const int dof = shared_->dof;
    const int nNo = lhs.nNo;
    const int mynNo = lhs.mynNo;

    FE_THROW_IF(dof <= 0, InvalidArgumentException, "FsilsVector::updateGhosts: invalid dof");
    FE_THROW_IF(nNo < 0, InvalidArgumentException, "FsilsVector::updateGhosts: invalid local node count");
    FE_THROW_IF(static_cast<int>(data_.size()) != dof * nNo,
                InvalidArgumentException, "FsilsVector::updateGhosts: local size mismatch");
    FE_THROW_IF(mynNo < 0 || mynNo > nNo,
                InvalidArgumentException, "FsilsVector::updateGhosts: invalid mynNo");

    // FSILS provides additive overlap communication. For a pure owner->ghost update, zero out
    // ghost slots so only the owning ranks contribute for each shared node.
    std::vector<double> u_internal(static_cast<std::size_t>(dof) * static_cast<std::size_t>(nNo), 0.0);
    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        for (int c = 0; c < dof; ++c) {
            const std::size_t old_idx = static_cast<std::size_t>(c) +
                                        static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
            const std::size_t int_idx = static_cast<std::size_t>(c) +
                                        static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof);
            u_internal[int_idx] = (internal < mynNo) ? data_[old_idx] : 0.0;
        }
    }

    Array<double> U(dof, nNo, u_internal.data());
    fe_fsi_linear_solver::fsils_commuv(lhs, dof, U);

    // Map back to old local ordering (owned + ghost).
    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        for (int c = 0; c < dof; ++c) {
            const std::size_t old_idx = static_cast<std::size_t>(c) +
                                        static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
            const std::size_t int_idx = static_cast<std::size_t>(c) +
                                        static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof);
            data_[old_idx] = u_internal[int_idx];
        }
    }
}

void FsilsVector::accumulateOverlap()
{
    if (!shared_) {
        return;
    }

    const auto& lhs = shared_->lhs;
    if (lhs.commu.nTasks == 1) {
        return;
    }

    const int dof = shared_->dof;
    const int nNo = lhs.nNo;

    FE_THROW_IF(dof <= 0, InvalidArgumentException, "FsilsVector::accumulateOverlap: invalid dof");
    FE_THROW_IF(nNo < 0, InvalidArgumentException, "FsilsVector::accumulateOverlap: invalid local node count");
    FE_THROW_IF(static_cast<int>(data_.size()) != dof * nNo,
                InvalidArgumentException, "FsilsVector::accumulateOverlap: local size mismatch");

    std::vector<double> u_internal(static_cast<std::size_t>(dof) * static_cast<std::size_t>(nNo), 0.0);
    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        for (int c = 0; c < dof; ++c) {
            const std::size_t old_idx = static_cast<std::size_t>(c) +
                                        static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
            const std::size_t int_idx = static_cast<std::size_t>(c) +
                                        static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof);
            u_internal[int_idx] = data_[old_idx];
        }
    }

    Array<double> U(dof, nNo, u_internal.data());
    fe_fsi_linear_solver::fsils_commuv(lhs, dof, U);

    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        for (int c = 0; c < dof; ++c) {
            const std::size_t old_idx = static_cast<std::size_t>(c) +
                                        static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
            const std::size_t int_idx = static_cast<std::size_t>(c) +
                                        static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof);
            data_[old_idx] = u_internal[int_idx];
        }
    }
}

std::unique_ptr<assembly::GlobalSystemView> FsilsVector::createAssemblyView()
{
    return std::make_unique<FsilsVectorView>(*this);
}

std::span<Real> FsilsVector::localSpan()
{
    return std::span<Real>(data_.data(), data_.size());
}

std::span<const Real> FsilsVector::localSpan() const
{
    return std::span<const Real>(data_.data(), data_.size());
}

} // namespace backends
} // namespace FE
} // namespace svmp
