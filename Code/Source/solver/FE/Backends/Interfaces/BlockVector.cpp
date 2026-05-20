/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Interfaces/BlockVector.h"

#include <cmath>

namespace svmp {
namespace FE {
namespace backends {

BlockVector::BlockVector(std::vector<std::unique_ptr<GenericVector>> blocks)
    : blocks_(std::move(blocks))
{
    FE_THROW_IF(blocks_.empty(), InvalidArgumentException, "BlockVector: at least one block is required");

    FE_CHECK_NOT_NULL(blocks_[0].get(), "BlockVector: block[0]");
    backend_kind_ = blocks_[0]->backendKind();

    offsets_.assign(blocks_.size() + 1, 0);
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        FE_CHECK_NOT_NULL(blocks_[i].get(), "BlockVector: block");
        FE_THROW_IF(blocks_[i]->backendKind() != backend_kind_, InvalidArgumentException,
                    "BlockVector: mixed backend kinds are not supported");
        offsets_[i + 1] = offsets_[i] + blocks_[i]->size();
    }
    global_size_ = offsets_.back();
}

std::uint64_t BlockVector::valueRevision() const noexcept
{
    std::uint64_t h = value_revision_;
    for (const auto& block : blocks_) {
        if (block) {
            const auto v = block->valueRevision();
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
        }
    }
    return h;
}

void BlockVector::markModified() noexcept
{
    ++value_revision_;
    for (auto& block : blocks_) {
        if (block) {
            block->markModified();
        }
    }
}

void BlockVector::zero()
{
    for (auto& b : blocks_) {
        b->zero();
    }
    ++value_revision_;
}

void BlockVector::set(Real value)
{
    for (auto& b : blocks_) {
        b->set(value);
    }
    ++value_revision_;
}

void BlockVector::add(Real value)
{
    for (auto& b : blocks_) {
        b->add(value);
    }
    ++value_revision_;
}

void BlockVector::scale(Real alpha)
{
    for (auto& b : blocks_) {
        b->scale(alpha);
    }
    ++value_revision_;
}

void BlockVector::copyFrom(const GenericVector& other)
{
    const auto* o = dynamic_cast<const BlockVector*>(&other);
    FE_THROW_IF(!o, InvalidArgumentException, "BlockVector::copyFrom: backend mismatch (expected BlockVector)");
    FE_THROW_IF(o->numBlocks() != numBlocks(), InvalidArgumentException, "BlockVector::copyFrom: block count mismatch");
    FE_THROW_IF(o->offsets_ != offsets_, InvalidArgumentException, "BlockVector::copyFrom: block layout mismatch");

    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        blocks_[i]->copyFrom(*o->blocks_[i]);
    }
    ++value_revision_;
}

Real BlockVector::dot(const GenericVector& other) const
{
    const auto* o = dynamic_cast<const BlockVector*>(&other);
    FE_THROW_IF(!o, InvalidArgumentException, "BlockVector::dot: backend mismatch (expected BlockVector)");
    FE_THROW_IF(o->numBlocks() != numBlocks(), InvalidArgumentException, "BlockVector::dot: block count mismatch");
    FE_THROW_IF(o->offsets_ != offsets_, InvalidArgumentException, "BlockVector::dot: block layout mismatch");

    Real sum = 0.0;
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        sum += blocks_[i]->dot(*o->blocks_[i]);
    }
    return sum;
}

Real BlockVector::norm() const
{
    return std::sqrt(dot(*this));
}

void BlockVector::updateGhosts()
{
    for (auto& b : blocks_) {
        b->updateGhosts();
    }
    ++value_revision_;
}

GenericVector& BlockVector::block(std::size_t i)
{
    FE_THROW_IF(i >= blocks_.size(), InvalidArgumentException, "BlockVector::block: index out of range");
    FE_CHECK_NOT_NULL(blocks_[i].get(), "BlockVector::block");
    return *blocks_[i];
}

const GenericVector& BlockVector::block(std::size_t i) const
{
    FE_THROW_IF(i >= blocks_.size(), InvalidArgumentException, "BlockVector::block: index out of range");
    FE_CHECK_NOT_NULL(blocks_[i].get(), "BlockVector::block");
    return *blocks_[i];
}

std::pair<std::size_t, GlobalIndex> BlockVector::locate(GlobalIndex global_index) const
{
    if (global_index < 0 || global_index >= global_size_) {
        return {blocks_.size(), -1};
    }

    const auto it = std::upper_bound(offsets_.begin(), offsets_.end(), global_index);
    if (it == offsets_.begin() || it == offsets_.end()) {
        return {blocks_.size(), -1};
    }

    const auto block_idx = static_cast<std::size_t>((it - offsets_.begin()) - 1);
    const auto local = global_index - offsets_[block_idx];
    return {block_idx, local};
}

namespace {

class BlockVectorView final : public assembly::GlobalSystemView {
public:
    explicit BlockVectorView(BlockVector& vec) : vec_(&vec)
    {
        FE_CHECK_NOT_NULL(vec_, "BlockVectorView::vec");
        block_views_.reserve(vec_->numBlocks());
        for (std::size_t i = 0; i < vec_->numBlocks(); ++i) {
            block_views_.push_back(vec_->block(i).createAssemblyView());
        }
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
        if (dofs.size() != local_vector.size()) {
            FE_THROW(InvalidArgumentException, "BlockVectorView::addVectorEntries: size mismatch");
        }
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            addVectorEntry(dofs[i], local_vector[i], mode);
        }
    }

    void addVectorEntry(GlobalIndex dof, Real value, assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(vec_, "BlockVectorView::vec");
        const auto [bidx, local] = vec_->locate(dof);
        if (bidx >= block_views_.size()) {
            return;
        }
        block_views_[bidx]->addVectorEntry(local, value, mode);
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
        FE_CHECK_NOT_NULL(vec_, "BlockVectorView::vec");
        const auto [bidx, local] = vec_->locate(dof);
        if (bidx >= block_views_.size()) {
            return 0.0;
        }
        return block_views_[bidx]->getVectorEntry(local);
    }

    void beginAssemblyPhase() override
    {
        for (auto& v : block_views_) {
            v->beginAssemblyPhase();
        }
        phase_ = assembly::AssemblyPhase::Building;
    }

    void endAssemblyPhase() override
    {
        for (auto& v : block_views_) {
            v->endAssemblyPhase();
        }
        phase_ = assembly::AssemblyPhase::Flushing;
    }

    void finalizeAssembly() override
    {
        for (auto& v : block_views_) {
            v->finalizeAssembly();
        }
        phase_ = assembly::AssemblyPhase::Finalized;
    }

    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return phase_; }

    [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
    [[nodiscard]] bool hasVector() const noexcept override { return true; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return vec_ ? vec_->size() : 0; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return 1; }

    [[nodiscard]] bool isDistributed() const noexcept override
    {
        for (const auto& v : block_views_) {
            if (v->isDistributed()) return true;
        }
        return false;
    }

    [[nodiscard]] std::string backendName() const override { return "BlockVector"; }

    void zero() override
    {
        FE_CHECK_NOT_NULL(vec_, "BlockVectorView::vec");
        vec_->zero();
    }

private:
    BlockVector* vec_{nullptr};
    std::vector<std::unique_ptr<assembly::GlobalSystemView>> block_views_{};
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
};

} // namespace

std::unique_ptr<assembly::GlobalSystemView> BlockVector::createAssemblyView()
{
    return std::make_unique<BlockVectorView>(*this);
}

std::span<Real> BlockVector::localSpan()
{
    FE_THROW_IF(blocks_.size() != 1, NotImplementedException,
                "BlockVector::localSpan: only supported for a single block");
    return blocks_[0]->localSpan();
}

std::span<const Real> BlockVector::localSpan() const
{
    FE_THROW_IF(blocks_.size() != 1, NotImplementedException,
                "BlockVector::localSpan: only supported for a single block");
    return blocks_[0]->localSpan();
}

} // namespace backends
} // namespace FE
} // namespace svmp
