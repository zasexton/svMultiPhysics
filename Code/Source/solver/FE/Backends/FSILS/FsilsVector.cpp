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
#include <mpi.h>
#include <string>

namespace svmp {
namespace FE {
namespace backends {

namespace {

	class FsilsVectorView final : public assembly::GlobalSystemView {
	public:
	    explicit FsilsVectorView(FsilsVector& vec) : vec_(&vec) {}

    // Matrix operations (no-op)
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

        const auto* shared = vec_->shared();
        if (!shared) {
            // No shared metadata: fall back to per-entry insertion.
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                addVectorEntry(dofs[i], local_vector[i], mode);
            }
            return;
        }

        const int dof_per_node = shared->dof;
        const auto perm = shared->dof_permutation;
        const bool have_perm = perm && !perm->empty();
        const GlobalIndex vec_size = vec_->size();
        auto& data = vec_->data();

        int cached_global_node = -1;
        int cached_old = -1;

        for (std::size_t i = 0; i < dofs.size(); ++i) {
            GlobalIndex dof_idx = dofs[i];
            if (dof_idx < 0 || dof_idx >= vec_size) continue;

            if (have_perm) {
                if (static_cast<std::size_t>(dof_idx) >= perm->forward.size()) continue;
                dof_idx = perm->forward[static_cast<std::size_t>(dof_idx)];
            }
            if (dof_idx < 0 || dof_idx >= vec_size) continue;

            const int global_node = static_cast<int>(dof_idx / dof_per_node);
            const int comp = static_cast<int>(dof_idx % dof_per_node);

            // Reuse cached old index if same node as previous DOF.
            int old;
            if (global_node == cached_global_node) {
                old = cached_old;
            } else {
                old = shared->globalNodeToOld(global_node);
                cached_global_node = global_node;
                cached_old = old;
            }
            if (old < 0) continue;

            const std::size_t idx = static_cast<std::size_t>(old) * static_cast<std::size_t>(dof_per_node) +
                                    static_cast<std::size_t>(comp);
            if (idx >= data.size()) continue;

            switch (mode) {
                case assembly::AddMode::Add:
                    data[idx] += local_vector[i];
                    break;
                case assembly::AddMode::Insert:
                    data[idx] = local_vector[i];
                    break;
                case assembly::AddMode::Max:
                    data[idx] = std::max(data[idx], local_vector[i]);
                    break;
                case assembly::AddMode::Min:
                    data[idx] = std::min(data[idx], local_vector[i]);
                    break;
            }
        }
    }

	    void addVectorEntry(GlobalIndex dof, Real value, assembly::AddMode mode) override
	    {
	        FE_CHECK_NOT_NULL(vec_, "FsilsVectorView::vec");
	        if (dof < 0 || dof >= vec_->size()) {
	            return;
	        }

		        if (const auto* shared = vec_->shared()) {
		            if (const auto perm = shared->dof_permutation; perm && !perm->empty()) {
		                const auto& fwd = perm->forward;
		                if (static_cast<std::size_t>(dof) >= fwd.size()) {
		                    return;
		                }
		                dof = fwd[static_cast<std::size_t>(dof)];
		            }
		        }
		        if (dof < 0 || dof >= vec_->size()) {
		            return;
		        }

		        std::size_t idx = 0;
		        if (const auto* shared = vec_->shared()) {
		            const int dof_per_node = shared->dof;
		            const int global_node = static_cast<int>(dof / dof_per_node);
	            const int comp = static_cast<int>(dof % dof_per_node);
            const int old = shared->globalNodeToOld(global_node);
            if (old < 0) return;
            idx = static_cast<std::size_t>(old) * static_cast<std::size_t>(dof_per_node) +
                  static_cast<std::size_t>(comp);
        } else {
            idx = static_cast<std::size_t>(dof);
        }

        auto& data = vec_->data();
        if (idx >= data.size()) return;
        switch (mode) {
            case assembly::AddMode::Add:
                data[idx] += value;
                break;
            case assembly::AddMode::Insert:
                data[idx] = value;
                break;
            case assembly::AddMode::Max:
                data[idx] = std::max(data[idx], value);
                break;
            case assembly::AddMode::Min:
                data[idx] = std::min(data[idx], value);
                break;
        }
    }

    void setVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> values) override
    {
        if (dofs.size() != values.size()) {
            FE_THROW(InvalidArgumentException, "FsilsVectorView::setVectorEntries: size mismatch");
        }
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            addVectorEntry(dofs[i], values[i], assembly::AddMode::Insert);
        }
    }

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override
    {
        for (const auto dof : dofs) {
            addVectorEntry(dof, 0.0, assembly::AddMode::Insert);
        }
    }

	    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override
	    {
	        FE_CHECK_NOT_NULL(vec_, "FsilsVectorView::vec");
	        if (dof < 0 || dof >= vec_->size()) {
	            return 0.0;
	        }

		        if (const auto* shared = vec_->shared()) {
		            if (const auto perm = shared->dof_permutation; perm && !perm->empty()) {
		                const auto& fwd = perm->forward;
		                if (static_cast<std::size_t>(dof) >= fwd.size()) {
		                    return 0.0;
		                }
		                dof = fwd[static_cast<std::size_t>(dof)];
		            }
		        }
		        if (dof < 0 || dof >= vec_->size()) {
		            return 0.0;
		        }

		        std::size_t idx = 0;
		        if (const auto* shared = vec_->shared()) {
		            const int dof_per_node = shared->dof;
		            const int global_node = static_cast<int>(dof / dof_per_node);
	            const int comp = static_cast<int>(dof % dof_per_node);
            const int old = shared->globalNodeToOld(global_node);
            if (old < 0) return 0.0;
            idx = static_cast<std::size_t>(old) * static_cast<std::size_t>(dof_per_node) +
                  static_cast<std::size_t>(comp);
        } else {
            idx = static_cast<std::size_t>(dof);
        }
        if (idx >= vec_->data().size()) return 0.0;
        return vec_->data()[idx];
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
