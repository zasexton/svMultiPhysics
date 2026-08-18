/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/PETSc/PetscVector.h"

#if defined(FE_HAS_PETSC)

#include "Core/FEException.h"

#include <algorithm>
#include <limits>

namespace svmp {
namespace FE {
namespace backends {

namespace {

[[nodiscard]] PetscInt asPetscInt(GlobalIndex v, const char* what)
{
    FE_THROW_IF(v < 0, InvalidArgumentException, std::string("PETSc: negative ") + what);
    FE_THROW_IF(v > static_cast<GlobalIndex>(std::numeric_limits<PetscInt>::max()),
                InvalidArgumentException,
                std::string("PETSc: ") + what + " exceeds PetscInt range");
    return static_cast<PetscInt>(v);
}

} // namespace

PetscVector::PetscVector(GlobalIndex global_size)
{
    const auto n = asPetscInt(global_size, "vector size");
    FE_PETSC_CALL(VecCreate(PETSC_COMM_WORLD, &vec_));
    FE_PETSC_CALL(VecSetSizes(vec_, PETSC_DECIDE, n));
    FE_PETSC_CALL(VecSetFromOptions(vec_));
    FE_PETSC_CALL(VecSet(vec_, 0.0));

    PetscInt n_local = 0;
    FE_PETSC_CALL(VecGetLocalSize(vec_, &n_local));
    local_owned_ = n_local;
    ghost_count_ = 0;
    ghosted_ = false;
}

PetscVector::PetscVector(GlobalIndex local_size, GlobalIndex global_size)
{
    const auto n_local = asPetscInt(local_size, "local vector size");
    const auto n_global = asPetscInt(global_size, "global vector size");

    FE_PETSC_CALL(VecCreate(PETSC_COMM_WORLD, &vec_));
    FE_PETSC_CALL(VecSetSizes(vec_, n_local, n_global));
    FE_PETSC_CALL(VecSetFromOptions(vec_));
    FE_PETSC_CALL(VecSet(vec_, 0.0));

    local_owned_ = n_local;
    ghost_count_ = 0;
    ghosted_ = false;
}

PetscVector::PetscVector(GlobalIndex local_size, GlobalIndex global_size, const std::vector<GlobalIndex>& ghost_global_indices)
{
    const auto n_local = asPetscInt(local_size, "local vector size");
    const auto n_global = asPetscInt(global_size, "global vector size");

    const auto n_ghost = asPetscInt(static_cast<GlobalIndex>(ghost_global_indices.size()), "ghost count");

    std::vector<PetscInt> ghosts;
    ghosts.reserve(ghost_global_indices.size());
    for (const auto g : ghost_global_indices) {
        const PetscInt gi = asPetscInt(g, "ghost index");
        FE_THROW_IF(gi >= n_global, InvalidArgumentException, "PETSc: ghost index out of range");
        ghosts.push_back(gi);
    }

    ghost_local_indices_.reserve(ghost_global_indices.size());
    for (std::size_t i = 0; i < ghost_global_indices.size(); ++i) {
        const auto [it, inserted] = ghost_local_indices_.emplace(
            ghost_global_indices[i],
            n_local + static_cast<PetscInt>(i));
        FE_THROW_IF(!inserted,
                    InvalidArgumentException,
                    "PETSc: duplicate ghost index " +
                        std::to_string(ghost_global_indices[i]));
        (void)it;
    }

    FE_PETSC_CALL(VecCreateGhost(PETSC_COMM_WORLD,
                                n_local,
                                n_global,
                                n_ghost,
                                ghosts.empty() ? nullptr : ghosts.data(),
                                &vec_));
    FE_PETSC_CALL(VecSetFromOptions(vec_));
    FE_PETSC_CALL(VecSet(vec_, 0.0));

    local_owned_ = n_local;
    ghost_count_ = n_ghost;
    // VecCreateGhost establishes a collective ghost-update layout even on a
    // rank with no local ghost leaves.  Such a rank can still own values that
    // peer ranks import, so it must participate in VecGhostUpdate.
    ghosted_ = true;
}

PetscVector::~PetscVector()
{
    if (vec_) {
        ensureVecUpToDate();
        FE_PETSC_CALL(VecDestroy(&vec_));
    }
}

PetscVector::PetscVector(PetscVector&& other) noexcept
{
    *this = std::move(other);
}

PetscVector& PetscVector::operator=(PetscVector&& other) noexcept
{
    if (this == &other) {
        return *this;
    }
    if (vec_) {
        // Best-effort cleanup; avoid throwing in noexcept.
        VecDestroy(&vec_);
    }
    local_owned_ = other.local_owned_;
    ghost_count_ = other.ghost_count_;
    ghosted_ = other.ghosted_;
    ghost_local_indices_ = std::move(other.ghost_local_indices_);
    vec_ = other.vec_;
    local_cache_ = std::move(other.local_cache_);
    local_cache_valid_ = other.local_cache_valid_;
    local_cache_dirty_ = other.local_cache_dirty_;
    value_revision_ = other.value_revision_;

    other.local_owned_ = 0;
    other.ghost_count_ = 0;
    other.ghosted_ = false;
    other.ghost_local_indices_.clear();
    other.vec_ = nullptr;
    other.local_cache_valid_ = false;
    other.local_cache_dirty_ = false;
    other.value_revision_ = 0;
    return *this;
}

GlobalIndex PetscVector::size() const noexcept
{
    if (!vec_) return 0;
    PetscInt n = 0;
    VecGetSize(vec_, &n);
    return static_cast<GlobalIndex>(n);
}

std::vector<GlobalIndex> PetscVector::ownedGlobalRows() const
{
    FE_THROW_IF(
        vec_ == nullptr,
        FEException,
        "PetscVector::ownedGlobalRows: vector is null");
    PetscInt first = 0;
    PetscInt last = 0;
    FE_PETSC_CALL(VecGetOwnershipRange(vec_, &first, &last));
    FE_THROW_IF(
        first < 0 || last < first || last - first != local_owned_ ||
            static_cast<GlobalIndex>(last) > size(),
        FEException,
        "PetscVector::ownedGlobalRows: PETSc ownership range is invalid");
    std::vector<GlobalIndex> rows;
    rows.reserve(static_cast<std::size_t>(last - first));
    for (PetscInt row = first; row < last; ++row) {
        rows.push_back(static_cast<GlobalIndex>(row));
    }
    return rows;
}

void PetscVector::ensureVecUpToDate() const
{
    if (!vec_) return;
    if (!local_cache_dirty_) return;

    PetscScalar* arr = nullptr;
    FE_PETSC_CALL(VecGetArray(vec_, &arr));
    const PetscInt n_owned = local_owned_;
    FE_THROW_IF(static_cast<std::size_t>(n_owned) > local_cache_.size(),
                FEException, "PETSc: local cache smaller than owned size");
    for (PetscInt i = 0; i < n_owned; ++i) {
        arr[i] = static_cast<PetscScalar>(local_cache_[static_cast<std::size_t>(i)]);
    }
    FE_PETSC_CALL(VecRestoreArray(vec_, &arr));

    local_cache_dirty_ = false;
}

void PetscVector::ensureCacheUpToDate() const
{
    if (!vec_) return;
    if (local_cache_valid_ && !local_cache_dirty_) return;

    ensureVecUpToDate();

    if (ghosted_ && ghost_count_ > 0) {
        Vec local = nullptr;
        FE_PETSC_CALL(VecGhostGetLocalForm(vec_, &local));

        PetscInt n_local = 0;
        FE_PETSC_CALL(VecGetLocalSize(local, &n_local));
        local_cache_.assign(static_cast<std::size_t>(n_local), 0.0);

        const PetscScalar* arr = nullptr;
        FE_PETSC_CALL(VecGetArrayRead(local, &arr));
        for (PetscInt i = 0; i < n_local; ++i) {
            local_cache_[static_cast<std::size_t>(i)] = static_cast<Real>(arr[i]);
        }
        FE_PETSC_CALL(VecRestoreArrayRead(local, &arr));
        FE_PETSC_CALL(VecGhostRestoreLocalForm(vec_, &local));
    } else {
        PetscInt n_local = 0;
        FE_PETSC_CALL(VecGetLocalSize(vec_, &n_local));
        local_cache_.assign(static_cast<std::size_t>(n_local), 0.0);

        const PetscScalar* arr = nullptr;
        FE_PETSC_CALL(VecGetArrayRead(vec_, &arr));
        for (PetscInt i = 0; i < n_local; ++i) {
            local_cache_[static_cast<std::size_t>(i)] = static_cast<Real>(arr[i]);
        }
        FE_PETSC_CALL(VecRestoreArrayRead(vec_, &arr));
    }

    local_cache_valid_ = true;
    local_cache_dirty_ = false;
}

void PetscVector::invalidateLocalCache() const noexcept
{
    local_cache_valid_ = false;
    local_cache_dirty_ = false;
    local_cache_.clear();
}

void PetscVector::zero()
{
    ensureVecUpToDate();
    FE_PETSC_CALL(VecSet(vec_, 0.0));
    invalidateLocalCache();
    markModified();
}

void PetscVector::set(Real value)
{
    ensureVecUpToDate();
    FE_PETSC_CALL(VecSet(vec_, static_cast<PetscScalar>(value)));
    invalidateLocalCache();
    markModified();
}

void PetscVector::add(Real value)
{
    ensureVecUpToDate();
    FE_PETSC_CALL(VecShift(vec_, static_cast<PetscScalar>(value)));
    invalidateLocalCache();
    markModified();
}

void PetscVector::scale(Real alpha)
{
    ensureVecUpToDate();
    FE_PETSC_CALL(VecScale(vec_, static_cast<PetscScalar>(alpha)));
    invalidateLocalCache();
    markModified();
}

void PetscVector::copyFrom(const GenericVector& other)
{
    const auto* o = dynamic_cast<const PetscVector*>(&other);
    FE_THROW_IF(!o, InvalidArgumentException, "PetscVector::copyFrom: backend mismatch");
    FE_THROW_IF(size() != o->size(), InvalidArgumentException, "PetscVector::copyFrom: size mismatch");
    o->ensureVecUpToDate();
    ensureVecUpToDate();
    FE_PETSC_CALL(VecCopy(o->vec_, vec_));
    invalidateLocalCache();
    markModified();
}

Real PetscVector::dot(const GenericVector& other) const
{
    const auto* o = dynamic_cast<const PetscVector*>(&other);
    FE_THROW_IF(!o, InvalidArgumentException, "PetscVector::dot: backend mismatch");
    ensureVecUpToDate();
    o->ensureVecUpToDate();

    PetscScalar v = 0.0;
    FE_PETSC_CALL(VecDot(vec_, o->vec_, &v));
    return static_cast<Real>(v);
}

Real PetscVector::norm() const
{
    ensureVecUpToDate();
    PetscReal n = 0.0;
    FE_PETSC_CALL(VecNorm(vec_, NORM_2, &n));
    return static_cast<Real>(n);
}

void PetscVector::updateGhosts()
{
    if (!vec_ || !ghosted_) {
        return;
    }

    ensureVecUpToDate();
    FE_PETSC_CALL(VecAssemblyBegin(vec_));
    FE_PETSC_CALL(VecAssemblyEnd(vec_));
    FE_PETSC_CALL(VecGhostUpdateBegin(vec_, INSERT_VALUES, SCATTER_FORWARD));
    FE_PETSC_CALL(VecGhostUpdateEnd(vec_, INSERT_VALUES, SCATTER_FORWARD));
    invalidateLocalCache();
    markModified();
}

void PetscVector::readGlobalEntries(
    std::span<const GlobalIndex> dofs,
    std::span<Real> values) const
{
    FE_THROW_IF(dofs.size() != values.size(),
                InvalidArgumentException,
                "PetscVector::readGlobalEntries: size mismatch");
    FE_THROW_IF(vec_ == nullptr,
                FEException,
                "PetscVector::readGlobalEntries: vector is null");

    ensureCacheUpToDate();
    PetscInt owned_first = 0;
    PetscInt owned_end = 0;
    FE_PETSC_CALL(VecGetOwnershipRange(vec_, &owned_first, &owned_end));
    FE_THROW_IF(owned_end - owned_first != local_owned_,
                FEException,
                "PetscVector::readGlobalEntries: ownership range changed");

    const auto global_size = size();
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        const auto dof = dofs[i];
        FE_THROW_IF(dof < 0 || dof >= global_size,
                    InvalidArgumentException,
                    "PetscVector::readGlobalEntries: global index out of range");

        PetscInt local = -1;
        if (dof >= static_cast<GlobalIndex>(owned_first) &&
            dof < static_cast<GlobalIndex>(owned_end)) {
            local = static_cast<PetscInt>(dof) - owned_first;
        } else if (const auto ghost = ghost_local_indices_.find(dof);
                   ghost != ghost_local_indices_.end()) {
            local = ghost->second;
        }

        FE_THROW_IF(local < 0 ||
                        static_cast<std::size_t>(local) >= local_cache_.size(),
                    FEException,
                    "PetscVector::readGlobalEntries: global index " +
                        std::to_string(dof) +
                        " is outside the owned/ghost read layout");
        values[i] = local_cache_[static_cast<std::size_t>(local)];
    }
}

namespace {

InsertMode toPetscInsertMode(assembly::AddMode mode)
{
    switch (mode) {
        case assembly::AddMode::Add: return ADD_VALUES;
        case assembly::AddMode::Insert: return INSERT_VALUES;
        case assembly::AddMode::Max: return MAX_VALUES;
        case assembly::AddMode::Min: return MIN_VALUES;
        default: return ADD_VALUES;
    }
}

class PetscVectorView final : public assembly::GlobalSystemView {
public:
    explicit PetscVectorView(PetscVector& vec,
                             bool locally_relevant_reads = false)
        : vec_(&vec), locally_relevant_reads_(locally_relevant_reads)
    {
    }

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
        FE_CHECK_NOT_NULL(vec_, "PetscVectorView::vec");
        if (dofs.size() != local_vector.size()) {
            FE_THROW(InvalidArgumentException, "PetscVectorView::addVectorEntries: size mismatch");
        }

        std::vector<PetscInt> idx(dofs.size());
        std::vector<PetscScalar> vals(dofs.size());
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            FE_THROW_IF(dofs[i] < 0, InvalidArgumentException, "PetscVectorView: negative dof index");
            idx[i] = static_cast<PetscInt>(dofs[i]);
            vals[i] = static_cast<PetscScalar>(local_vector[i]);
        }

        FE_PETSC_CALL(VecSetValues(vec_->petsc(),
                                   static_cast<PetscInt>(idx.size()),
                                   idx.data(),
                                   vals.data(),
                                   toPetscInsertMode(mode)));
        vec_->invalidateLocalCache();
        vec_->markModified();
    }

    void addVectorEntry(GlobalIndex dof, Real value, assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(vec_, "PetscVectorView::vec");
        FE_THROW_IF(dof < 0, InvalidArgumentException, "PetscVectorView: negative dof index");
        const PetscInt idx = static_cast<PetscInt>(dof);
        const PetscScalar v = static_cast<PetscScalar>(value);
        FE_PETSC_CALL(VecSetValues(vec_->petsc(), 1, &idx, &v, toPetscInsertMode(mode)));
        vec_->invalidateLocalCache();
        vec_->markModified();
    }

    void setVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> values) override
    {
        addVectorEntries(dofs, values, assembly::AddMode::Insert);
    }

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override
    {
        std::vector<Real> zeros(dofs.size(), 0.0);
        addVectorEntries(dofs, zeros, assembly::AddMode::Insert);
    }

    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override
    {
        FE_CHECK_NOT_NULL(vec_, "PetscVectorView::vec");
        if (!locally_relevant_reads_) {
            FE_THROW_IF(dof < 0,
                        InvalidArgumentException,
                        "PetscVectorView: negative dof index");
            const PetscInt idx = static_cast<PetscInt>(dof);
            PetscScalar value = 0.0;
            FE_PETSC_CALL(VecGetValues(
                vec_->petsc(), 1, &idx, &value));
            return static_cast<Real>(value);
        }
        Real value = 0.0;
        vec_->readGlobalEntries(
            std::span<const GlobalIndex>(&dof, 1u),
            std::span<Real>(&value, 1u));
        return value;
    }

    void getVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<Real> values) const override
    {
        FE_CHECK_NOT_NULL(vec_, "PetscVectorView::vec");
        FE_THROW_IF(dofs.size() != values.size(),
                    InvalidArgumentException,
                    "PetscVectorView::getVectorEntries: size mismatch");
        if (locally_relevant_reads_) {
            vec_->readGlobalEntries(dofs, values);
            return;
        }
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            values[i] = getVectorEntry(dofs[i]);
        }
    }

    void beginAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Building; }

    void endAssemblyPhase() override
    {
        FE_CHECK_NOT_NULL(vec_, "PetscVectorView::vec");
        FE_PETSC_CALL(VecAssemblyBegin(vec_->petsc()));
        FE_PETSC_CALL(VecAssemblyEnd(vec_->petsc()));
        phase_ = assembly::AssemblyPhase::Flushing;
    }

    void finalizeAssembly() override
    {
        // VecAssembly is idempotent; treat finalize as a final flush.
        endAssemblyPhase();
        phase_ = assembly::AssemblyPhase::Finalized;
    }

    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return phase_; }

    [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
    [[nodiscard]] bool hasVector() const noexcept override { return true; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return vec_ ? vec_->size() : 0; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return 1; }
    [[nodiscard]] bool isDistributed() const noexcept override { return true; }
    [[nodiscard]] std::string backendName() const override { return "PETScVector"; }

    void zero() override
    {
        FE_CHECK_NOT_NULL(vec_, "PetscVectorView::vec");
        vec_->zero();
    }

private:
    PetscVector* vec_{nullptr};
    bool locally_relevant_reads_{false};
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
};

} // namespace

std::unique_ptr<assembly::GlobalSystemView> PetscVector::createAssemblyView()
{
    return std::make_unique<PetscVectorView>(*this);
}

std::unique_ptr<assembly::GlobalSystemView>
PetscVector::createGhostedReadView()
{
    return std::make_unique<PetscVectorView>(
        *this, /*locally_relevant_reads=*/true);
}

std::span<Real> PetscVector::localSpan()
{
    ensureCacheUpToDate();
    local_cache_dirty_ = true;
    markModified();
    return std::span<Real>(local_cache_.data(), local_cache_.size());
}

std::span<const Real> PetscVector::localSpan() const
{
    ensureCacheUpToDate();
    return std::span<const Real>(local_cache_.data(), local_cache_.size());
}

Vec PetscVector::petsc() const
{
    ensureVecUpToDate();
    return vec_;
}

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_PETSC
