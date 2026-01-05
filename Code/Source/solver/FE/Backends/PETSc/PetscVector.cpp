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
    ghosted_ = (n_ghost > 0);
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
    vec_ = other.vec_;
    local_cache_ = std::move(other.local_cache_);
    local_cache_valid_ = other.local_cache_valid_;
    local_cache_dirty_ = other.local_cache_dirty_;

    other.local_owned_ = 0;
    other.ghost_count_ = 0;
    other.ghosted_ = false;
    other.vec_ = nullptr;
    other.local_cache_valid_ = false;
    other.local_cache_dirty_ = false;
    return *this;
}

GlobalIndex PetscVector::size() const noexcept
{
    if (!vec_) return 0;
    PetscInt n = 0;
    VecGetSize(vec_, &n);
    return static_cast<GlobalIndex>(n);
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
}

void PetscVector::set(Real value)
{
    ensureVecUpToDate();
    FE_PETSC_CALL(VecSet(vec_, static_cast<PetscScalar>(value)));
    invalidateLocalCache();
}

void PetscVector::add(Real value)
{
    ensureVecUpToDate();
    FE_PETSC_CALL(VecShift(vec_, static_cast<PetscScalar>(value)));
    invalidateLocalCache();
}

void PetscVector::scale(Real alpha)
{
    ensureVecUpToDate();
    FE_PETSC_CALL(VecScale(vec_, static_cast<PetscScalar>(alpha)));
    invalidateLocalCache();
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
    if (!vec_ || !ghosted_ || ghost_count_ == 0) {
        return;
    }

    ensureVecUpToDate();
    FE_PETSC_CALL(VecAssemblyBegin(vec_));
    FE_PETSC_CALL(VecAssemblyEnd(vec_));
    FE_PETSC_CALL(VecGhostUpdateBegin(vec_, INSERT_VALUES, SCATTER_FORWARD));
    FE_PETSC_CALL(VecGhostUpdateEnd(vec_, INSERT_VALUES, SCATTER_FORWARD));
    invalidateLocalCache();
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
    explicit PetscVectorView(PetscVector& vec) : vec_(&vec) {}

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
    }

    void addVectorEntry(GlobalIndex dof, Real value, assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(vec_, "PetscVectorView::vec");
        FE_THROW_IF(dof < 0, InvalidArgumentException, "PetscVectorView: negative dof index");
        const PetscInt idx = static_cast<PetscInt>(dof);
        const PetscScalar v = static_cast<PetscScalar>(value);
        FE_PETSC_CALL(VecSetValues(vec_->petsc(), 1, &idx, &v, toPetscInsertMode(mode)));
        vec_->invalidateLocalCache();
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
        FE_THROW_IF(dof < 0, InvalidArgumentException, "PetscVectorView: negative dof index");
        const PetscInt idx = static_cast<PetscInt>(dof);
        PetscScalar v = 0.0;
        FE_PETSC_CALL(VecGetValues(vec_->petsc(), 1, &idx, &v));
        return static_cast<Real>(v);
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
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
};

} // namespace

std::unique_ptr<assembly::GlobalSystemView> PetscVector::createAssemblyView()
{
    return std::make_unique<PetscVectorView>(*this);
}

std::span<Real> PetscVector::localSpan()
{
    ensureCacheUpToDate();
    local_cache_dirty_ = true;
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
