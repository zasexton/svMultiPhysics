/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_DEFERREDINSERTBUFFER_H
#define SVMP_FE_ASSEMBLY_DEFERREDINSERTBUFFER_H

/**
 * @file DeferredInsertBuffer.h
 * @brief Recorded global insertions of the threaded assembly path.
 *
 * In threaded assembly (FE/Docs/ThreadedAssembly.md) the threads compute local
 * element/face/cut-volume outputs but do not touch the global system. Each
 * call that the serial loop makes to one of StandardAssembler's insertion
 * routines is recorded here instead, with a copy of the local output and of
 * the row/column DOF lists. The assembler later replays the records of all
 * items in the original item order through the same routines, so the global
 * matrix and vector receive exactly the serial sequence of additions.
 */

#include "Core/Types.h"
#include "Assembly/AssemblyKernel.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace svmp {
namespace FE {

namespace dofs {
class DofMap;
}

namespace assembly {

class GlobalSystemView;

namespace detail {

/// One recorded call of a StandardAssembler insertion routine.
struct DeferredInsertOp {
    enum class Kind : std::uint8_t {
        ForCell,       ///< insertLocalForCell(cell, maps, output, rows, cols, views)
        Local,         ///< insertLocal(output, rows, cols, views) without resolved slots
        Constrained,   ///< insertLocalConstrained(output, rows, cols, views)
        MatrixEntries  ///< matrix_view->addMatrixEntries(rows, cols, output.local_matrix)
    };

    Kind kind{Kind::Local};
    GlobalIndex cell_id{-1};
    const dofs::DofMap* row_dof_map{nullptr};
    GlobalIndex row_dof_offset{0};
    const dofs::DofMap* col_dof_map{nullptr};
    GlobalIndex col_dof_offset{0};
    GlobalSystemView* matrix_view{nullptr};
    GlobalSystemView* vector_view{nullptr};
    std::size_t output_index{0};
    std::size_t rows_begin{0};
    std::size_t rows_size{0};
    std::size_t cols_begin{0};
    std::size_t cols_size{0};
};

/**
 * @brief Insertions recorded by one thread for one block of items, in order.
 *
 * Storage is kept between uses (clear() keeps capacity), so steady-state
 * recording does not allocate.  Aligned to a cache line: adjacent buffers of
 * the ring are written by different threads at the same time.
 */
class alignas(64) DeferredInsertBuffer {
public:
    void clear() noexcept
    {
        ops_.clear();
        indices_.clear();
        used_outputs_ = 0;
    }

    [[nodiscard]] bool empty() const noexcept { return ops_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return ops_.size(); }

    void record(DeferredInsertOp::Kind kind,
                const KernelOutput& output,
                std::span<const GlobalIndex> rows,
                std::span<const GlobalIndex> cols,
                GlobalSystemView* matrix_view,
                GlobalSystemView* vector_view,
                GlobalIndex cell_id = -1,
                const dofs::DofMap* row_dof_map = nullptr,
                GlobalIndex row_dof_offset = 0,
                const dofs::DofMap* col_dof_map = nullptr,
                GlobalIndex col_dof_offset = 0)
    {
        DeferredInsertOp op;
        op.kind = kind;
        op.cell_id = cell_id;
        op.row_dof_map = row_dof_map;
        op.row_dof_offset = row_dof_offset;
        op.col_dof_map = col_dof_map;
        op.col_dof_offset = col_dof_offset;
        op.matrix_view = matrix_view;
        op.vector_view = vector_view;
        if (used_outputs_ == outputs_.size()) {
            outputs_.emplace_back();
        }
        op.output_index = used_outputs_;
        outputs_[used_outputs_++] = output;
        op.rows_begin = indices_.size();
        op.rows_size = rows.size();
        indices_.insert(indices_.end(), rows.begin(), rows.end());
        op.cols_begin = indices_.size();
        op.cols_size = cols.size();
        indices_.insert(indices_.end(), cols.begin(), cols.end());
        ops_.push_back(op);
    }

    [[nodiscard]] const std::vector<DeferredInsertOp>& ops() const noexcept { return ops_; }

    [[nodiscard]] const KernelOutput& output(const DeferredInsertOp& op) const noexcept
    {
        return outputs_[op.output_index];
    }

    [[nodiscard]] std::span<const GlobalIndex> rows(const DeferredInsertOp& op) const noexcept
    {
        return std::span<const GlobalIndex>(indices_.data() + op.rows_begin, op.rows_size);
    }

    [[nodiscard]] std::span<const GlobalIndex> cols(const DeferredInsertOp& op) const noexcept
    {
        return std::span<const GlobalIndex>(indices_.data() + op.cols_begin, op.cols_size);
    }

private:
    std::vector<DeferredInsertOp> ops_{};
    std::vector<KernelOutput> outputs_{};
    std::size_t used_outputs_{0};
    std::vector<GlobalIndex> indices_{};
};

} // namespace detail
} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_DEFERREDINSERTBUFFER_H
