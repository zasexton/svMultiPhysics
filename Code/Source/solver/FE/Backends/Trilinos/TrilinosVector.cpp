/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Trilinos/TrilinosVector.h"

#if defined(FE_HAS_TRILINOS)

#include "Core/FEException.h"

#include <Teuchos_OrdinalTraits.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace backends {

namespace {

[[nodiscard]] trilinos::GO asGo(GlobalIndex v, const char* what)
{
    FE_THROW_IF(v < 0, InvalidArgumentException, std::string("Trilinos: negative ") + what);
    return static_cast<trilinos::GO>(v);
}

} // namespace

TrilinosVector::TrilinosVector(GlobalIndex global_size)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto n_global = static_cast<Tpetra::global_size_t>(asGo(global_size, "vector size"));
    map_ = Teuchos::rcp(new trilinos::Map(n_global, 0, comm));
    vec_ = Teuchos::rcp(new trilinos::Vector(map_));
    vec_->putScalar(0.0);
}

TrilinosVector::TrilinosVector(GlobalIndex local_size, GlobalIndex global_size)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto n_global = static_cast<Tpetra::global_size_t>(asGo(global_size, "global vector size"));
    const auto n_local = static_cast<std::size_t>(asGo(local_size, "local vector size"));
    map_ = Teuchos::rcp(new trilinos::Map(n_global, n_local, 0, comm));
    vec_ = Teuchos::rcp(new trilinos::Vector(map_));
    vec_->putScalar(0.0);
}

TrilinosVector::TrilinosVector(GlobalIndex owned_first,
                               GlobalIndex local_owned_size,
                               GlobalIndex global_size,
                               const std::vector<GlobalIndex>& ghost_global_indices)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto n_global = static_cast<Tpetra::global_size_t>(asGo(global_size, "global vector size"));
    const auto n_local = static_cast<std::size_t>(asGo(local_owned_size, "local vector size"));

    map_ = Teuchos::rcp(new trilinos::Map(n_global, n_local, 0, comm));
    vec_ = Teuchos::rcp(new trilinos::Vector(map_));
    vec_->putScalar(0.0);

    if (!ghost_global_indices.empty()) {
        std::vector<trilinos::GO> overlap_gids;
        overlap_gids.reserve(n_local + ghost_global_indices.size());

        FE_THROW_IF(owned_first < 0, InvalidArgumentException, "TrilinosVector: negative owned_first");
        for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(n_local); ++i) {
            overlap_gids.push_back(asGo(owned_first + i, "owned gid"));
        }
        for (const auto g : ghost_global_indices) {
            overlap_gids.push_back(asGo(g, "ghost gid"));
        }

        overlap_map_ = Teuchos::rcp(new trilinos::Map(n_global,
                                                      Teuchos::ArrayView<const trilinos::GO>(overlap_gids.data(),
                                                                                           static_cast<int>(overlap_gids.size())),
                                                      0,
                                                      comm));
        overlap_vec_ = Teuchos::rcp(new trilinos::Vector(overlap_map_));
        overlap_vec_->putScalar(0.0);

        importer_ = Teuchos::rcp(new Tpetra::Import<trilinos::LO, trilinos::GO, trilinos::Node>(map_, overlap_map_));
    }
}

GlobalIndex TrilinosVector::size() const noexcept
{
    if (vec_.is_null()) return 0;
    return static_cast<GlobalIndex>(vec_->getGlobalLength());
}

void TrilinosVector::syncVectorFromCache() const
{
    if (vec_.is_null()) return;
    if (!local_cache_dirty_) return;

    auto data = vec_->getDataNonConst(0);
    const std::size_t n_owned = static_cast<std::size_t>(data.size());
    FE_THROW_IF(local_cache_.size() < n_owned, FEException, "TrilinosVector: cache smaller than owned span");
    for (std::size_t i = 0; i < n_owned; ++i) {
        data[static_cast<trilinos::LO>(i)] = static_cast<trilinos::Scalar>(local_cache_[i]);
    }
    local_cache_dirty_ = false;
}

void TrilinosVector::syncCacheFromVector() const
{
    if (vec_.is_null()) return;
    if (local_cache_valid_ && !local_cache_dirty_) return;

    syncVectorFromCache();

    const auto owned_data = vec_->getData(0);
    const std::size_t n_owned = static_cast<std::size_t>(owned_data.size());

    std::size_t n_total = n_owned;
    if (!overlap_vec_.is_null()) {
        n_total = static_cast<std::size_t>(overlap_vec_->getLocalLength());
        FE_THROW_IF(n_total < n_owned, FEException, "TrilinosVector: overlap vector smaller than owned vector");
    }

    local_cache_.assign(n_total, 0.0);
    for (std::size_t i = 0; i < n_owned; ++i) {
        local_cache_[i] = static_cast<Real>(owned_data[static_cast<trilinos::LO>(i)]);
    }

    if (!overlap_vec_.is_null() && n_total > n_owned) {
        const auto overlap_data = overlap_vec_->getData(0);
        FE_THROW_IF(static_cast<std::size_t>(overlap_data.size()) != n_total, FEException,
                    "TrilinosVector: unexpected overlap data size");
        for (std::size_t i = n_owned; i < n_total; ++i) {
            local_cache_[i] = static_cast<Real>(overlap_data[static_cast<trilinos::LO>(i)]);
        }
    }

    local_cache_valid_ = true;
    local_cache_dirty_ = false;
}

void TrilinosVector::invalidateLocalCache() const noexcept
{
    local_cache_.clear();
    local_cache_valid_ = false;
    local_cache_dirty_ = false;
}

void TrilinosVector::zero()
{
    syncVectorFromCache();
    vec_->putScalar(0.0);
    invalidateLocalCache();
}

void TrilinosVector::set(Real value)
{
    syncVectorFromCache();
    vec_->putScalar(static_cast<trilinos::Scalar>(value));
    invalidateLocalCache();
}

void TrilinosVector::add(Real value)
{
    syncVectorFromCache();
    auto data = vec_->getDataNonConst(0);
    for (trilinos::LO i = 0; i < data.size(); ++i) {
        data[i] += static_cast<trilinos::Scalar>(value);
    }
    invalidateLocalCache();
}

void TrilinosVector::scale(Real alpha)
{
    syncVectorFromCache();
    vec_->scale(static_cast<trilinos::Scalar>(alpha));
    invalidateLocalCache();
}

Real TrilinosVector::dot(const GenericVector& other) const
{
    const auto* o = dynamic_cast<const TrilinosVector*>(&other);
    FE_THROW_IF(!o, InvalidArgumentException, "TrilinosVector::dot: backend mismatch");
    syncVectorFromCache();
    o->syncVectorFromCache();
    return static_cast<Real>(vec_->dot(*o->vec_));
}

Real TrilinosVector::norm() const
{
    syncVectorFromCache();
    return static_cast<Real>(vec_->norm2());
}

void TrilinosVector::updateGhosts()
{
    if (vec_.is_null() || overlap_vec_.is_null() || importer_.is_null()) {
        return;
    }

    syncVectorFromCache();
    overlap_vec_->doImport(*vec_, *importer_, Tpetra::INSERT);
    invalidateLocalCache();
}

namespace {

class TrilinosVectorView final : public assembly::GlobalSystemView {
public:
    explicit TrilinosVectorView(TrilinosVector& vec) : vec_(&vec) {}

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
            FE_THROW(InvalidArgumentException, "TrilinosVectorView::addVectorEntries: size mismatch");
        }
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            addVectorEntry(dofs[i], local_vector[i], mode);
        }
    }

    void addVectorEntry(GlobalIndex dof, Real value, assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(vec_, "TrilinosVectorView::vec");
        if (dof < 0 || dof >= vec_->size()) {
            return;
        }

        const auto gid = static_cast<trilinos::GO>(dof);
        const auto lid = vec_->map()->getLocalElement(gid);
        if (lid == Teuchos::OrdinalTraits<trilinos::LO>::invalid()) {
            FE_THROW(NotImplementedException,
                     "TrilinosVectorView::addVectorEntry: nonlocal entry insertion is not supported; "
                     "assemble owned entries only or use PETSc for off-process insertion");
        }

        auto data = vec_->tpetra()->getDataNonConst(0);
        const auto idx = static_cast<trilinos::LO>(lid);

        switch (mode) {
            case assembly::AddMode::Add:
                data[idx] += static_cast<trilinos::Scalar>(value);
                break;
            case assembly::AddMode::Insert:
                data[idx] = static_cast<trilinos::Scalar>(value);
                break;
            case assembly::AddMode::Max:
                data[idx] = std::max(data[idx], static_cast<trilinos::Scalar>(value));
                break;
            case assembly::AddMode::Min:
                data[idx] = std::min(data[idx], static_cast<trilinos::Scalar>(value));
                break;
        }

        vec_->invalidateLocalCache();
    }

    void setVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> values) override
    {
        addVectorEntries(dofs, values, assembly::AddMode::Insert);
    }

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override
    {
        for (const auto d : dofs) {
            addVectorEntry(d, 0.0, assembly::AddMode::Insert);
        }
    }

    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override
    {
        FE_CHECK_NOT_NULL(vec_, "TrilinosVectorView::vec");
        if (dof < 0) return 0.0;
        const auto lid = vec_->map()->getLocalElement(static_cast<trilinos::GO>(dof));
        if (lid == Teuchos::OrdinalTraits<trilinos::LO>::invalid()) {
            return 0.0;
        }
        const auto idx = static_cast<trilinos::LO>(lid);
        const auto data = vec_->tpetra()->getData(0);
        if (idx < 0 || idx >= data.size()) return 0.0;
        return static_cast<Real>(data[idx]);
    }

    void beginAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Building; }
    void endAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Flushing; }
    void finalizeAssembly() override { phase_ = assembly::AssemblyPhase::Finalized; }
    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return phase_; }

    [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
    [[nodiscard]] bool hasVector() const noexcept override { return true; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return vec_ ? vec_->size() : 0; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return 1; }
    [[nodiscard]] bool isDistributed() const noexcept override { return true; }
    [[nodiscard]] std::string backendName() const override { return "TrilinosVector"; }

    void zero() override
    {
        FE_CHECK_NOT_NULL(vec_, "TrilinosVectorView::vec");
        vec_->zero();
    }

private:
    TrilinosVector* vec_{nullptr};
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
};

} // namespace

std::unique_ptr<assembly::GlobalSystemView> TrilinosVector::createAssemblyView()
{
    return std::make_unique<TrilinosVectorView>(*this);
}

std::span<Real> TrilinosVector::localSpan()
{
    syncCacheFromVector();
    local_cache_dirty_ = true;
    return std::span<Real>(local_cache_.data(), local_cache_.size());
}

std::span<const Real> TrilinosVector::localSpan() const
{
    syncCacheFromVector();
    return std::span<const Real>(local_cache_.data(), local_cache_.size());
}

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_TRILINOS
