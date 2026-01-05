/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Interfaces/BlockMatrix.h"

#include <span>

namespace svmp {
namespace FE {
namespace backends {

namespace {

[[nodiscard]] std::vector<GlobalIndex> prefixOffsets(const std::vector<GlobalIndex>& sizes, const char* what)
{
    FE_THROW_IF(sizes.empty(), InvalidArgumentException, std::string("BlockMatrix: empty ") + what);
    std::vector<GlobalIndex> offsets(sizes.size() + 1, 0);
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        FE_THROW_IF(sizes[i] < 0, InvalidArgumentException, std::string("BlockMatrix: negative ") + what);
        offsets[i + 1] = offsets[i] + sizes[i];
    }
    return offsets;
}

[[nodiscard]] std::pair<std::size_t, GlobalIndex> locateInOffsets(std::span<const GlobalIndex> offsets,
                                                                  GlobalIndex idx) noexcept
{
    if (offsets.empty() || idx < 0 || idx >= offsets.back()) {
        return {offsets.size(), -1};
    }
    const auto it = std::upper_bound(offsets.begin(), offsets.end(), idx);
    if (it == offsets.begin() || it == offsets.end()) {
        return {offsets.size(), -1};
    }
    const auto block_idx = static_cast<std::size_t>((it - offsets.begin()) - 1);
    const auto local = idx - offsets[block_idx];
    return {block_idx, local};
}

} // namespace

BlockMatrix::BlockMatrix(std::vector<GlobalIndex> row_block_sizes,
                         std::vector<GlobalIndex> col_block_sizes,
                         std::vector<std::vector<std::unique_ptr<GenericMatrix>>> blocks)
    : row_block_sizes_(std::move(row_block_sizes)),
      col_block_sizes_(std::move(col_block_sizes)),
      row_offsets_(prefixOffsets(row_block_sizes_, "row block sizes")),
      col_offsets_(prefixOffsets(col_block_sizes_, "col block sizes")),
      blocks_(std::move(blocks))
{
    const auto m = row_block_sizes_.size();
    const auto n = col_block_sizes_.size();

    FE_THROW_IF(blocks_.size() != m, InvalidArgumentException, "BlockMatrix: block grid row count mismatch");
    for (std::size_t i = 0; i < m; ++i) {
        FE_THROW_IF(blocks_[i].size() != n, InvalidArgumentException, "BlockMatrix: block grid col count mismatch");
    }

    n_rows_ = row_offsets_.back();
    n_cols_ = col_offsets_.back();

    bool backend_set = false;
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const auto* blk = blocks_[i][j].get();
            if (!blk) continue;

            FE_THROW_IF(blk->numRows() != row_block_sizes_[i] || blk->numCols() != col_block_sizes_[j],
                        InvalidArgumentException, "BlockMatrix: block dimension mismatch");

            if (!backend_set) {
                backend_kind_ = blk->backendKind();
                backend_set = true;
            } else {
                FE_THROW_IF(blk->backendKind() != backend_kind_, InvalidArgumentException,
                            "BlockMatrix: mixed backend kinds are not supported");
            }
        }
    }

    FE_THROW_IF(!backend_set, InvalidArgumentException, "BlockMatrix: at least one non-null block is required");
}

void BlockMatrix::zero()
{
    for (auto& row : blocks_) {
        for (auto& blk : row) {
            if (blk) blk->zero();
        }
    }
}

void BlockMatrix::finalizeAssembly()
{
    for (auto& row : blocks_) {
        for (auto& blk : row) {
            if (blk) blk->finalizeAssembly();
        }
    }
}

void BlockMatrix::mult(const GenericVector& x, GenericVector& y) const
{
    auto* yb = dynamic_cast<BlockVector*>(&y);
    FE_THROW_IF(!yb, InvalidArgumentException, "BlockMatrix::mult: y must be a BlockVector");
    yb->zero();
    multAdd(x, y);
}

void BlockMatrix::multAdd(const GenericVector& x, GenericVector& y) const
{
    const auto* xb = dynamic_cast<const BlockVector*>(&x);
    auto* yb = dynamic_cast<BlockVector*>(&y);
    FE_THROW_IF(!xb || !yb, InvalidArgumentException, "BlockMatrix::multAdd: x/y must be BlockVector");
    FE_THROW_IF(xb->numBlocks() != numColBlocks(), InvalidArgumentException, "BlockMatrix::multAdd: x block mismatch");
    FE_THROW_IF(yb->numBlocks() != numRowBlocks(), InvalidArgumentException, "BlockMatrix::multAdd: y block mismatch");

    for (std::size_t i = 0; i < numRowBlocks(); ++i) {
        FE_THROW_IF(yb->block(i).size() != row_block_sizes_[i], InvalidArgumentException,
                    "BlockMatrix::multAdd: y block size mismatch");
    }
    for (std::size_t j = 0; j < numColBlocks(); ++j) {
        FE_THROW_IF(xb->block(j).size() != col_block_sizes_[j], InvalidArgumentException,
                    "BlockMatrix::multAdd: x block size mismatch");
    }

    for (std::size_t i = 0; i < numRowBlocks(); ++i) {
        for (std::size_t j = 0; j < numColBlocks(); ++j) {
            const auto* Aij = blocks_[i][j].get();
            if (!Aij) continue;
            Aij->multAdd(xb->block(j), yb->block(i));
        }
    }
}

Real BlockMatrix::getEntry(GlobalIndex row, GlobalIndex col) const
{
    const auto [ri, lr] = locateRow(row);
    const auto [cj, lc] = locateCol(col);
    if (ri >= numRowBlocks() || cj >= numColBlocks()) {
        return 0.0;
    }
    const auto* blk = block(ri, cj);
    if (!blk) return 0.0;
    return blk->getEntry(lr, lc);
}

GlobalIndex BlockMatrix::rowBlockOffset(std::size_t i) const
{
    FE_THROW_IF(i >= row_block_sizes_.size(), InvalidArgumentException, "BlockMatrix::rowBlockOffset: out of range");
    return row_offsets_[i];
}

GlobalIndex BlockMatrix::colBlockOffset(std::size_t j) const
{
    FE_THROW_IF(j >= col_block_sizes_.size(), InvalidArgumentException, "BlockMatrix::colBlockOffset: out of range");
    return col_offsets_[j];
}

GlobalIndex BlockMatrix::rowBlockSize(std::size_t i) const
{
    FE_THROW_IF(i >= row_block_sizes_.size(), InvalidArgumentException, "BlockMatrix::rowBlockSize: out of range");
    return row_block_sizes_[i];
}

GlobalIndex BlockMatrix::colBlockSize(std::size_t j) const
{
    FE_THROW_IF(j >= col_block_sizes_.size(), InvalidArgumentException, "BlockMatrix::colBlockSize: out of range");
    return col_block_sizes_[j];
}

GenericMatrix* BlockMatrix::block(std::size_t i, std::size_t j) noexcept
{
    if (i >= blocks_.size() || blocks_.empty() || j >= blocks_[i].size()) return nullptr;
    return blocks_[i][j].get();
}

const GenericMatrix* BlockMatrix::block(std::size_t i, std::size_t j) const noexcept
{
    if (i >= blocks_.size() || blocks_.empty() || j >= blocks_[i].size()) return nullptr;
    return blocks_[i][j].get();
}

std::pair<std::size_t, GlobalIndex> BlockMatrix::locateRow(GlobalIndex row) const
{
    return locateInOffsets(row_offsets_, row);
}

std::pair<std::size_t, GlobalIndex> BlockMatrix::locateCol(GlobalIndex col) const
{
    return locateInOffsets(col_offsets_, col);
}

namespace {

class BlockMatrixView final : public assembly::GlobalSystemView {
public:
    explicit BlockMatrixView(BlockMatrix& mat) : mat_(&mat)
    {
        FE_CHECK_NOT_NULL(mat_, "BlockMatrixView::mat");
        const auto m = mat_->numRowBlocks();
        const auto n = mat_->numColBlocks();
        views_.resize(m);
        for (std::size_t i = 0; i < m; ++i) {
            views_[i].resize(n);
            for (std::size_t j = 0; j < n; ++j) {
                if (auto* blk = mat_->block(i, j)) {
                    views_[i][j] = blk->createAssemblyView();
                }
            }
        }
    }

    void addMatrixEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_matrix,
                          assembly::AddMode mode) override
    {
        addMatrixEntries(dofs, dofs, local_matrix, mode);
    }

    void addMatrixEntries(std::span<const GlobalIndex> row_dofs,
                          std::span<const GlobalIndex> col_dofs,
                          std::span<const Real> local_matrix,
                          assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(mat_, "BlockMatrixView::mat");
        const GlobalIndex n_rows = static_cast<GlobalIndex>(row_dofs.size());
        const GlobalIndex n_cols = static_cast<GlobalIndex>(col_dofs.size());

        if (local_matrix.size() != static_cast<std::size_t>(n_rows * n_cols)) {
            FE_THROW(InvalidArgumentException, "BlockMatrixView::addMatrixEntries: local_matrix size mismatch");
        }

        for (GlobalIndex i = 0; i < n_rows; ++i) {
            const GlobalIndex row = row_dofs[static_cast<std::size_t>(i)];
            if (row < 0 || row >= mat_->numRows()) continue;

            for (GlobalIndex j = 0; j < n_cols; ++j) {
                const GlobalIndex col = col_dofs[static_cast<std::size_t>(j)];
                if (col < 0 || col >= mat_->numCols()) continue;

                const auto local_idx = static_cast<std::size_t>(i * n_cols + j);
                addMatrixEntry(row, col, local_matrix[local_idx], mode);
            }
        }
    }

    void addMatrixEntry(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(mat_, "BlockMatrixView::mat");
        const auto [rb, lr] = mat_->locateRow(row);
        const auto [cb, lc] = mat_->locateCol(col);
        if (rb >= views_.size() || cb >= views_[rb].size()) {
            return;
        }
        auto& v = views_[rb][cb];
        if (!v) return;
        v->addMatrixEntry(lr, lc, value, mode);
    }

    void setDiagonal(std::span<const GlobalIndex> dofs,
                     std::span<const Real> values) override
    {
        if (dofs.size() != values.size()) {
            FE_THROW(InvalidArgumentException, "BlockMatrixView::setDiagonal: size mismatch");
        }
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            setDiagonal(dofs[i], values[i]);
        }
    }

    void setDiagonal(GlobalIndex dof, Real value) override
    {
        addMatrixEntry(dof, dof, value, assembly::AddMode::Insert);
    }

    void zeroRows(std::span<const GlobalIndex> rows, bool set_diagonal) override
    {
        FE_CHECK_NOT_NULL(mat_, "BlockMatrixView::mat");
        for (const auto row : rows) {
            const auto [rb, lr] = mat_->locateRow(row);
            if (rb >= views_.size()) continue;

            for (std::size_t cb = 0; cb < views_[rb].size(); ++cb) {
                auto& v = views_[rb][cb];
                if (!v) continue;

                const bool diag = set_diagonal && (rb == cb);
                const GlobalIndex local_row = lr;
                v->zeroRows(std::span<const GlobalIndex>(&local_row, 1), diag);
            }
        }
    }

    // Vector ops (no-op)
    void addVectorEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
    void addVectorEntry(GlobalIndex, Real, assembly::AddMode) override {}
    void setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) override {}
    void zeroVectorEntries(std::span<const GlobalIndex>) override {}

    void beginAssemblyPhase() override
    {
        for (auto& row : views_) {
            for (auto& v : row) {
                if (v) v->beginAssemblyPhase();
            }
        }
        phase_ = assembly::AssemblyPhase::Building;
    }

    void endAssemblyPhase() override
    {
        for (auto& row : views_) {
            for (auto& v : row) {
                if (v) v->endAssemblyPhase();
            }
        }
        phase_ = assembly::AssemblyPhase::Flushing;
    }

    void finalizeAssembly() override
    {
        for (auto& row : views_) {
            for (auto& v : row) {
                if (v) v->finalizeAssembly();
            }
        }
        phase_ = assembly::AssemblyPhase::Finalized;
    }

    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return phase_; }

    [[nodiscard]] bool hasMatrix() const noexcept override { return true; }
    [[nodiscard]] bool hasVector() const noexcept override { return false; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return mat_ ? mat_->numRows() : 0; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return mat_ ? mat_->numCols() : 0; }

    [[nodiscard]] bool isDistributed() const noexcept override
    {
        for (const auto& row : views_) {
            for (const auto& v : row) {
                if (v && v->isDistributed()) return true;
            }
        }
        return false;
    }

    [[nodiscard]] std::string backendName() const override { return "BlockMatrix"; }

    void zero() override
    {
        FE_CHECK_NOT_NULL(mat_, "BlockMatrixView::mat");
        mat_->zero();
    }

    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override
    {
        FE_CHECK_NOT_NULL(mat_, "BlockMatrixView::mat");
        return mat_->getEntry(row, col);
    }

private:
    BlockMatrix* mat_{nullptr};
    std::vector<std::vector<std::unique_ptr<assembly::GlobalSystemView>>> views_{};
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
};

} // namespace

std::unique_ptr<assembly::GlobalSystemView> BlockMatrix::createAssemblyView()
{
    return std::make_unique<BlockMatrixView>(*this);
}

} // namespace backends
} // namespace FE
} // namespace svmp
