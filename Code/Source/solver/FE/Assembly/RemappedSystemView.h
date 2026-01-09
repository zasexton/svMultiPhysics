/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_REMAPPED_SYSTEM_VIEW_H
#define SVMP_FE_ASSEMBLY_REMAPPED_SYSTEM_VIEW_H

/**
 * @file RemappedSystemView.h
 * @brief GlobalSystemView decorator for RIS-style DOF remap + duplicate assembly
 *
 * This decorator supports assembly patterns where local element contributions
 * must be added to an additional (mapped) set of global DOFs without changing
 * the underlying DOF numbering. This is used to reproduce legacy behaviors
 * like `doassem_ris()` (duplicate assembly into an adjacent/mapped DOF set),
 * while keeping the FE library physics-agnostic.
 */

#include "Assembly/GlobalSystemView.h"

#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {

/**
 * @brief Abstract DOF remapper used by RemappedSystemView
 *
 * Return `std::nullopt` when no remap applies.
 */
class DofRemapper {
public:
    virtual ~DofRemapper() = default;
    [[nodiscard]] virtual std::optional<GlobalIndex> map(GlobalIndex dof) const noexcept = 0;
};

/**
 * @brief Simple hash-table remapper (GlobalIndex -> GlobalIndex)
 */
class DofRemapTable final : public DofRemapper {
public:
    void set(GlobalIndex from, GlobalIndex to) { table_[from] = to; }

    [[nodiscard]] std::optional<GlobalIndex> map(GlobalIndex dof) const noexcept override
    {
        const auto it = table_.find(dof);
        if (it == table_.end()) return std::nullopt;
        return it->second;
    }

private:
    std::unordered_map<GlobalIndex, GlobalIndex> table_{};
};

/**
 * @brief GlobalSystemView decorator that duplicates local contributions into mapped DOFs
 *
 * Semantics:
 * - Always inserts the original local contributions into the underlying view.
 * - For each local *row* DOF that has a mapping, an additional row is inserted
 *   into the mapped row DOF.
 * - Columns are mapped when possible (if the column DOF itself has a mapping),
 *   otherwise the original column DOF is used.
 *
 * This matches the legacy `doassem_ris()` behavior: only mapped *rows* trigger
 * additional assembly; mapped columns are applied opportunistically.
 *
 * Note: This wrapper is not thread-safe (it allocates scratch buffers per call).
 */
class RemappedSystemView final : public GlobalSystemView {
public:
    RemappedSystemView(GlobalSystemView& base, const DofRemapper& remapper)
        : base_(&base), remapper_(&remapper)
    {
    }

    RemappedSystemView(RemappedSystemView&& other) noexcept
        : base_(other.base_), remapper_(other.remapper_)
    {
        other.base_ = nullptr;
        other.remapper_ = nullptr;
    }

    RemappedSystemView& operator=(RemappedSystemView&& other) noexcept
    {
        if (this != &other) {
            base_ = other.base_;
            remapper_ = other.remapper_;
            other.base_ = nullptr;
            other.remapper_ = nullptr;
        }
        return *this;
    }

    // Matrix operations
    void addMatrixEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_matrix,
                          AddMode mode = AddMode::Add) override
    {
        addMatrixEntries(dofs, dofs, local_matrix, mode);
    }

    void addMatrixEntries(std::span<const GlobalIndex> row_dofs,
                          std::span<const GlobalIndex> col_dofs,
                          std::span<const Real> local_matrix,
                          AddMode mode = AddMode::Add) override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        FE_CHECK_NOT_NULL(remapper_, "RemappedSystemView::remapper");

        const std::size_t n_rows = row_dofs.size();
        const std::size_t n_cols = col_dofs.size();
        FE_THROW_IF(local_matrix.size() != n_rows * n_cols, InvalidArgumentException,
                    "RemappedSystemView::addMatrixEntries: local_matrix size mismatch");

        // Always assemble the original contribution.
        base_->addMatrixEntries(row_dofs, col_dofs, local_matrix, mode);

        // Duplicate into mapped rows.
        std::vector<GlobalIndex> mapped_cols(n_cols);
        std::vector<Real> row_values(n_cols);

        for (std::size_t i = 0; i < n_rows; ++i) {
            const GlobalIndex row = row_dofs[i];
            if (row < 0) continue;

            const auto mapped_row_opt = remapper_->map(row);
            if (!mapped_row_opt.has_value()) continue;
            const GlobalIndex mapped_row = *mapped_row_opt;
            if (mapped_row < 0 || mapped_row == row) continue;

            for (std::size_t j = 0; j < n_cols; ++j) {
                const GlobalIndex col = col_dofs[j];
                GlobalIndex out_col = col;
                if (col >= 0) {
                    const auto mapped_col_opt = remapper_->map(col);
                    if (mapped_col_opt.has_value() && mapped_col_opt.value() >= 0) {
                        out_col = mapped_col_opt.value();
                    }
                }
                mapped_cols[j] = out_col;
                row_values[j] = local_matrix[i * n_cols + j];
            }

            const GlobalIndex mapped_row_buf[1] = {mapped_row};
            base_->addMatrixEntries(mapped_row_buf, mapped_cols, row_values, mode);
        }
    }

    void addMatrixEntry(GlobalIndex row,
                        GlobalIndex col,
                        Real value,
                        AddMode mode = AddMode::Add) override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        FE_CHECK_NOT_NULL(remapper_, "RemappedSystemView::remapper");

        base_->addMatrixEntry(row, col, value, mode);
        if (row < 0) return;

        const auto mapped_row_opt = remapper_->map(row);
        if (!mapped_row_opt.has_value()) return;
        const GlobalIndex mapped_row = *mapped_row_opt;
        if (mapped_row < 0 || mapped_row == row) return;

        GlobalIndex out_col = col;
        if (col >= 0) {
            const auto mapped_col_opt = remapper_->map(col);
            if (mapped_col_opt.has_value() && mapped_col_opt.value() >= 0) {
                out_col = mapped_col_opt.value();
            }
        }
        base_->addMatrixEntry(mapped_row, out_col, value, mode);
    }

    void setDiagonal(std::span<const GlobalIndex> dofs,
                     std::span<const Real> values) override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        base_->setDiagonal(dofs, values);
    }

    void setDiagonal(GlobalIndex dof, Real value) override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        base_->setDiagonal(dof, value);
    }

    void zeroRows(std::span<const GlobalIndex> rows, bool set_diagonal = true) override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        base_->zeroRows(rows, set_diagonal);
    }

    // Vector operations
    void addVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_vector,
                          AddMode mode = AddMode::Add) override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        FE_CHECK_NOT_NULL(remapper_, "RemappedSystemView::remapper");
        FE_THROW_IF(dofs.size() != local_vector.size(), InvalidArgumentException,
                    "RemappedSystemView::addVectorEntries: size mismatch");

        base_->addVectorEntries(dofs, local_vector, mode);

        for (std::size_t i = 0; i < dofs.size(); ++i) {
            const GlobalIndex dof = dofs[i];
            if (dof < 0) continue;
            const auto mapped_opt = remapper_->map(dof);
            if (!mapped_opt.has_value()) continue;
            const GlobalIndex mapped = *mapped_opt;
            if (mapped < 0 || mapped == dof) continue;
            base_->addVectorEntry(mapped, local_vector[i], mode);
        }
    }

    void addVectorEntry(GlobalIndex dof, Real value, AddMode mode = AddMode::Add) override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        FE_CHECK_NOT_NULL(remapper_, "RemappedSystemView::remapper");

        base_->addVectorEntry(dof, value, mode);
        if (dof < 0) return;

        const auto mapped_opt = remapper_->map(dof);
        if (!mapped_opt.has_value()) return;
        const GlobalIndex mapped = *mapped_opt;
        if (mapped < 0 || mapped == dof) return;
        base_->addVectorEntry(mapped, value, mode);
    }

    void setVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> values) override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        base_->setVectorEntries(dofs, values);
    }

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        base_->zeroVectorEntries(dofs);
    }

    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        return base_->getVectorEntry(dof);
    }

    // Assembly lifecycle
    void beginAssemblyPhase() override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        base_->beginAssemblyPhase();
    }

    void endAssemblyPhase() override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        base_->endAssemblyPhase();
    }

    void finalizeAssembly() override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        base_->finalizeAssembly();
    }

    [[nodiscard]] AssemblyPhase getPhase() const noexcept override
    {
        return base_ ? base_->getPhase() : AssemblyPhase::NotStarted;
    }

    // Properties
    [[nodiscard]] bool hasMatrix() const noexcept override { return base_ && base_->hasMatrix(); }
    [[nodiscard]] bool hasVector() const noexcept override { return base_ && base_->hasVector(); }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return base_ ? base_->numRows() : 0; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return base_ ? base_->numCols() : 0; }
    [[nodiscard]] std::string backendName() const override { return base_ ? base_->backendName() : "RemappedSystem"; }

    // Zero and access
    void zero() override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        base_->zero();
    }

    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override
    {
        FE_CHECK_NOT_NULL(base_, "RemappedSystemView::base");
        return base_->getMatrixEntry(row, col);
    }

private:
    GlobalSystemView* base_{nullptr};
    const DofRemapper* remapper_{nullptr};
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_REMAPPED_SYSTEM_VIEW_H
