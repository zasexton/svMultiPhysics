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
    static int trace_budget = 0;
    static bool trace_init = false;
    if (!trace_init) {
        trace_init = true;
        if (std::getenv("SVMP_MONO_AUX_TRACE") != nullptr) {
            trace_budget = 100000;
        }
    }

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        const auto input_dof = dofs[i];
        auto dof = input_dof;
        if (input_dof < 0 || input_dof >= vec_size) {
            resolved[i] = INVALID_GLOBAL_INDEX;
            continue;
        }

        auto fs_dof = dof;
        if (has_perm) {
            if (static_cast<std::size_t>(fs_dof) >= fwd_size) {
                resolved[i] = INVALID_GLOBAL_INDEX;
                continue;
            }
            fs_dof = fwd_data[static_cast<std::size_t>(fs_dof)];
            if (fs_dof < 0 || fs_dof >= vec_size) {
                resolved[i] = INVALID_GLOBAL_INDEX;
                continue;
            }
        }
        dof = fs_dof;

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

        const bool trace_selected =
            (input_dof >= 7200 && input_dof < 8400);
        if (trace_budget > 0 && trace_selected) {
            const int internal_node = shared->globalNodeToInternal(global_node);
            GlobalIndex inverse_dof = INVALID_GLOBAL_INDEX;
            if (has_perm && perm_ptr->inverse.size() > static_cast<std::size_t>(fs_dof)) {
                inverse_dof = perm_ptr->inverse[static_cast<std::size_t>(fs_dof)];
            }
            std::fprintf(stderr,
                         "[FsilsVectorResolve] fe_dof=%lld fs_dof=%lld inverse_fe=%lld global_node=%d comp=%d old_node=%d internal_node=%d shnNo=%d mynNo=%d nNo=%d old_global=%d resolved=%lld\n",
                         static_cast<long long>(input_dof),
                         static_cast<long long>(fs_dof),
                         static_cast<long long>(inverse_dof),
                         global_node,
                         comp,
                         old_node,
                         internal_node,
                         shared->lhs.shnNo,
                         shared->lhs.mynNo,
                         shared->lhs.nNo,
                         shared->oldToGlobalNode(old_node),
                         static_cast<long long>(resolved[i]));
            --trace_budget;
        }
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
    static int trace_budget = -1;
    if (trace_budget < 0) {
        trace_budget = (std::getenv("SVMP_MONO_AUX_TRACE") != nullptr) ? 128 : 0;
    }
    const auto* shared = vec.shared();
    const int dof = shared ? shared->dof : 1;
    for (std::size_t i = 0; i < resolved.size(); ++i) {
        const auto idx = resolved[i];
        out[i] = (idx >= 0 && idx < data_size) ? data[static_cast<std::size_t>(idx)] : 0.0;
        if (trace_budget > 0 && shared && idx >= 280 && idx <= 1280) {
            const int old_node = static_cast<int>(idx / dof);
            const int comp = static_cast<int>(idx % dof);
            const int internal = (old_node >= 0 && old_node < shared->lhs.nNo) ? shared->lhs.map(old_node) : -1;
            const GlobalIndex internal_idx =
                (internal >= 0)
                    ? static_cast<GlobalIndex>(internal) * static_cast<GlobalIndex>(dof) + static_cast<GlobalIndex>(comp)
                    : INVALID_GLOBAL_INDEX;
            const Real internal_value =
                (internal_idx >= 0 && internal_idx < data_size) ? data[static_cast<std::size_t>(internal_idx)] : 0.0;
            std::fprintf(stderr,
                         "[FsilsVectorGather] resolved=%lld old_node=%d comp=%d internal=%d shnNo=%d mynNo=%d nNo=%d old_value=%.17g internal_value=%.17g\n",
                         static_cast<long long>(idx),
                         old_node,
                         comp,
                         internal,
                         shared->lhs.shnNo,
                         shared->lhs.mynNo,
                         shared->lhs.nNo,
                         static_cast<double>(out[i]),
                         static_cast<double>(internal_value));
            --trace_budget;
        }
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

void exchangeFsilsOwnedHalo(const FsilsShared& shared,
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

    FE_THROW_IF(dof <= 0, InvalidArgumentException, "FsilsVector owned-halo exchange: invalid dof");
    FE_THROW_IF(nNo < 0, InvalidArgumentException, "FsilsVector owned-halo exchange: invalid local node count");
    FE_THROW_IF(static_cast<int>(data.size()) != dof * nNo,
                InvalidArgumentException, "FsilsVector owned-halo exchange: local size mismatch");
    FE_THROW_IF(mynNo < 0 || mynNo > nNo,
                InvalidArgumentException, "FsilsVector owned-halo exchange: invalid mynNo");

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
            internal_work[int_idx] = static_cast<double>(data[old_idx]);
        }
    }

    Array<double> U(dof, nNo, internal_work.data());
    FE_THROW_IF(!lhs.owned_row_operator,
                InvalidArgumentException,
                "FsilsVector: FE FSILS vectors require explicit owned-row layout");
    if (owner_to_ghost_only) {
        fe_fsi_linear_solver::fsils_syncv_owned_to_ghost(lhs, dof, U);
    } else {
        fe_fsi_linear_solver::fsils_reverse_scatterv_contribution_buffer(lhs, dof, U);
        fe_fsi_linear_solver::fsils_syncv_owned_to_ghost(lhs, dof, U);
    }

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
        vector_requested_ += resolved.size();
        for (std::size_t i = 0; i < resolved.size(); ++i) {
            const auto idx = resolved[i];
            if (idx >= 0 && static_cast<std::size_t>(idx) < vec_->data().size()) {
                ++vector_valid_;
                accumulateVectorValueStats(local_vector[i]);
            }
        }
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
        vector_requested_ += resolved.size();
        for (std::size_t i = 0; i < resolved.size(); ++i) {
            const auto idx = resolved[i];
            if (idx >= 0 && static_cast<std::size_t>(idx) < vec_->data().size()) {
                ++vector_valid_;
                accumulateVectorValueStats(local_vector[i]);
            }
        }
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

        ++vector_requested_;
        ++vector_valid_;
        accumulateVectorValueStats(value);

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
    void finalizeAssembly() override {
        phase_ = assembly::AssemblyPhase::Finalized;
        if (std::getenv("SVMP_FSILS_VECTOR_VIEW_TRACE") != nullptr && vector_requested_ > 0) {
            int rank = 0;
            if (vec_ != nullptr) {
                if (const auto* shared = vec_->shared()) {
                    rank = shared->lhs.commu.task;
                }
            }
            std::fprintf(stderr,
                         "[FsilsVectorView] rank=%d requested=%zu valid=%zu invalid=%zu nonzero=%zu value_l1=%.17g value_l2=%.17g value_max_abs=%.17g size=%zu\n",
                         rank,
                         vector_requested_,
                         vector_valid_,
                         vector_requested_ - vector_valid_,
                         vector_nonzero_,
                         static_cast<double>(vector_value_l1_),
                         static_cast<double>(std::sqrt(vector_value_l2_sq_)),
                         static_cast<double>(vector_max_abs_),
                         vec_ ? vec_->data().size() : std::size_t(0));
        }
    }
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
    void accumulateVectorValueStats(Real value) noexcept
    {
        const Real abs_value = std::abs(value);
        vector_value_l1_ += abs_value;
        vector_value_l2_sq_ += value * value;
        vector_max_abs_ = std::max(vector_max_abs_, abs_value);
        if (abs_value > 0.0) {
            ++vector_nonzero_;
        }
    }

    FsilsVector* vec_{nullptr};
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
    std::size_t vector_requested_{0};
    std::size_t vector_valid_{0};
    std::size_t vector_nonzero_{0};
    Real vector_value_l1_{0.0};
    Real vector_value_l2_sq_{0.0};
    Real vector_max_abs_{0.0};
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

void FsilsVector::exchangeOwnedHalo(bool owner_to_ghost_only)
{
    if (!shared_) {
        return;
    }

    exchangeFsilsOwnedHalo(*shared_, data_, halo_internal_work_, owner_to_ghost_only);
}

void FsilsVector::updateGhosts()
{
    exchangeOwnedHalo(/*owner_to_ghost_only=*/true);
}

void FsilsVector::accumulateRawContributionsAndUpdateGhosts()
{
    exchangeOwnedHalo(/*owner_to_ghost_only=*/false);
}

bool FsilsVector::usesOwnedRowLayout() const noexcept
{
    return shared_ != nullptr && shared_->lhs.owned_row_operator;
}

bool FsilsVector::ownsFeDof(GlobalIndex fe_dof) const noexcept
{
    if (fe_dof < 0 || fe_dof >= global_size_) {
        return false;
    }
    if (!shared_) {
        return true;
    }
    if (shared_->dof <= 0) {
        return false;
    }

    GlobalIndex backend_dof = fe_dof;
    if (const auto perm = shared_->dof_permutation; perm && !perm->empty()) {
        if (static_cast<std::size_t>(fe_dof) >= perm->forward.size()) {
            return false;
        }
        backend_dof = perm->forward[static_cast<std::size_t>(fe_dof)];
    }
    if (backend_dof < 0 || backend_dof >= global_size_) {
        return false;
    }

    const int global_node = static_cast<int>(backend_dof / shared_->dof);
    const int old = shared_->globalNodeToOld(global_node);
    if (old < 0 || old >= shared_->lhs.nNo) {
        return false;
    }

    return old < shared_->owned_node_count;
}

std::vector<GlobalIndex> FsilsVector::ownedFeDofs() const
{
    if (!shared_) {
        std::vector<GlobalIndex> out;
        out.reserve(static_cast<std::size_t>(std::max<GlobalIndex>(global_size_, 0)));
        for (GlobalIndex dof = 0; dof < global_size_; ++dof) {
            out.push_back(dof);
        }
        return out;
    }

    const int dof = shared_->dof;
    FE_THROW_IF(dof <= 0, InvalidArgumentException, "FsilsVector::ownedFeDofs: invalid dof");

    const auto* perm = shared_->dof_permutation.get();
    const bool has_inverse = perm != nullptr && !perm->inverse.empty();
    const int owned_nodes = shared_->owned_node_count;

    std::vector<GlobalIndex> out;
    out.reserve(static_cast<std::size_t>(std::max(owned_nodes, 0)) *
                static_cast<std::size_t>(dof));

    for (int old = 0; old < shared_->lhs.nNo; ++old) {
        if (old >= shared_->owned_node_count) {
            continue;
        }

        const int backend_node = shared_->oldToGlobalNode(old);
        if (backend_node < 0) {
            continue;
        }
        for (int c = 0; c < dof; ++c) {
            const GlobalIndex backend_dof =
                static_cast<GlobalIndex>(backend_node) * static_cast<GlobalIndex>(dof) +
                static_cast<GlobalIndex>(c);
            if (backend_dof < 0 || backend_dof >= global_size_) {
                continue;
            }

            GlobalIndex fe_dof = backend_dof;
            if (has_inverse) {
                if (static_cast<std::size_t>(backend_dof) >= perm->inverse.size()) {
                    continue;
                }
                fe_dof = perm->inverse[static_cast<std::size_t>(backend_dof)];
            }
            if (fe_dof >= 0 && fe_dof < global_size_) {
                out.push_back(fe_dof);
            }
        }
    }

    return out;
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
