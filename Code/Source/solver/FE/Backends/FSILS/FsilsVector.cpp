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

void applyResolvedVectorEntries(std::vector<Real>& data,
                                std::span<const GlobalIndex> resolved,
                                std::span<const Real> local_vector,
                                assembly::AddMode mode)
{
    FE_THROW_IF(resolved.size() != local_vector.size(), InvalidArgumentException,
                "applyResolvedVectorEntries: size mismatch");

    const auto data_size = static_cast<GlobalIndex>(data.size());
    const auto* slots = resolved.data();
    const auto* local = local_vector.data();
    const std::size_t n = resolved.size();
    constexpr std::size_t kMinContiguousRun = 2u;

    const auto apply_contiguous_runs =
        [&](auto&& op_scalar, auto&& op_run) {
            std::size_t idx = 0;
            while (idx < n) {
                const auto slot = slots[idx];
                if (slot < 0 || slot >= data_size) {
                    ++idx;
                    continue;
                }

                std::size_t run = 1u;
                while (idx + run < n) {
                    const auto next_slot = slots[idx + run];
                    if (next_slot < 0 ||
                        next_slot >= data_size ||
                        next_slot != slot + static_cast<GlobalIndex>(run)) {
                        break;
                    }
                    ++run;
                }

                if (run >= kMinContiguousRun) {
                    op_run(data.data() + static_cast<std::size_t>(slot), local + idx, run);
                    idx += run;
                } else {
                    op_scalar(data[static_cast<std::size_t>(slot)], local[idx]);
                    ++idx;
                }
            }
        };

    switch (mode) {
        case assembly::AddMode::Add:
            apply_contiguous_runs(
                [](Real& dst, Real src) { dst += src; },
                [](Real* dst, const Real* src, std::size_t count) {
                    for (std::size_t k = 0; k < count; ++k) {
                        dst[k] += src[k];
                    }
                });
            break;
        case assembly::AddMode::Insert:
            apply_contiguous_runs(
                [](Real& dst, Real src) { dst = src; },
                [](Real* dst, const Real* src, std::size_t count) {
                    std::copy_n(src, count, dst);
                });
            break;
        case assembly::AddMode::Max:
            apply_contiguous_runs(
                [](Real& dst, Real src) { dst = std::max(dst, src); },
                [](Real* dst, const Real* src, std::size_t count) {
                    for (std::size_t k = 0; k < count; ++k) {
                        dst[k] = std::max(dst[k], src[k]);
                    }
                });
            break;
        case assembly::AddMode::Min:
            apply_contiguous_runs(
                [](Real& dst, Real src) { dst = std::min(dst, src); },
                [](Real* dst, const Real* src, std::size_t count) {
                    for (std::size_t k = 0; k < count; ++k) {
                        dst[k] = std::min(dst[k], src[k]);
                    }
                });
            break;
    }
}

void exchangeFsilsOverlap(const FsilsShared& shared,
                          std::vector<Real>& data,
                          std::vector<double>& internal_work,
                          bool owner_to_ghost_only)
{
    const auto& lhs = shared.lhs;
    if (lhs.commu.nTasks == 1) {
        return;
    }

    const int dof = shared.dof;
    const int nNo = lhs.nNo;
    const int mynNo = lhs.mynNo;

    FE_THROW_IF(dof <= 0, InvalidArgumentException, "FsilsVector overlap exchange: invalid dof");
    FE_THROW_IF(nNo < 0, InvalidArgumentException, "FsilsVector overlap exchange: invalid local node count");
    FE_THROW_IF(static_cast<int>(data.size()) != dof * nNo,
                InvalidArgumentException, "FsilsVector overlap exchange: local size mismatch");
    FE_THROW_IF(mynNo < 0 || mynNo > nNo,
                InvalidArgumentException, "FsilsVector overlap exchange: invalid mynNo");

    const auto work_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(nNo);
    internal_work.resize(work_size);
    std::fill(internal_work.begin(), internal_work.end(), 0.0);

    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        for (int c = 0; c < dof; ++c) {
            const std::size_t old_idx = static_cast<std::size_t>(c) +
                                        static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
            const std::size_t int_idx = static_cast<std::size_t>(c) +
                                        static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof);
            internal_work[int_idx] =
                (owner_to_ghost_only && internal >= mynNo) ? 0.0 : static_cast<double>(data[old_idx]);
        }
    }

    Array<double> U(dof, nNo, internal_work.data());
    fe_fsi_linear_solver::fsils_commuv(lhs, dof, U);

    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        for (int c = 0; c < dof; ++c) {
            const std::size_t old_idx = static_cast<std::size_t>(c) +
                                        static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
            const std::size_t int_idx = static_cast<std::size_t>(c) +
                                        static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof);
            data[old_idx] = static_cast<Real>(internal_work[int_idx]);
        }
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
        applyResolvedVectorEntries(vec_->data(),
                                   std::span<const GlobalIndex>(resolved),
                                   local_vector,
                                   mode);
    }

    void addVectorEntriesResolved(std::span<const GlobalIndex> dofs,
                                   std::span<const GlobalIndex> resolved,
                                   std::span<const Real> local_vector,
                                   assembly::AddMode mode) override
    {
        (void)dofs;
        FE_CHECK_NOT_NULL(vec_, "FsilsVectorView::vec");
        applyResolvedVectorEntries(vec_->data(), resolved, local_vector, mode);
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

    [[nodiscard]] assembly::InsertionCapabilities insertionCapabilities() const noexcept override
    {
        return assembly::InsertionCapabilities{
            .resolved_matrix_entries = false,
            .resolved_vector_entries = vectorLayoutHandle() != nullptr,
            .contiguous_combined_matrix_insert = false,
            .exact_rank_one_updates = false,
        };
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
            auto& commu = const_cast<fe_fsi_linear_solver::FSILS_commuType&>(lhs.commu);
            fe_fsi_linear_solver::fsils_allreduce_sum(&sum, &global_sum, 1, MPI_DOUBLE, commu);
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

void FsilsVector::exchangeOverlap(bool owner_to_ghost_only)
{
    if (!shared_) {
        return;
    }

    exchangeFsilsOverlap(*shared_, data_, overlap_internal_work_, owner_to_ghost_only);
}

void FsilsVector::updateGhosts()
{
    exchangeOverlap(/*owner_to_ghost_only=*/true);
}

void FsilsVector::accumulateOverlap()
{
    exchangeOverlap(/*owner_to_ghost_only=*/false);
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
