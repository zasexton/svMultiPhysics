/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "BlockAssembler.h"
#include "Core/FEException.h"

#include <chrono>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// BlockView Implementation
// ============================================================================

BlockView::BlockView(GlobalSystemView& global_view,
                     GlobalIndex row_offset, GlobalIndex col_offset,
                     GlobalIndex num_rows, GlobalIndex num_cols)
    : global_view_(global_view)
    , row_offset_(row_offset)
    , col_offset_(col_offset)
    , num_rows_(num_rows)
    , num_cols_(num_cols)
{
}

void BlockView::addMatrixEntries(
    std::span<const GlobalIndex> local_rows,
    std::span<const GlobalIndex> local_cols,
    std::span<const Real> values)
{
    // Translate local indices to global
    translated_rows_.resize(local_rows.size());
    translated_cols_.resize(local_cols.size());

    for (std::size_t i = 0; i < local_rows.size(); ++i) {
        translated_rows_[i] = local_rows[i] + row_offset_;
    }
    for (std::size_t i = 0; i < local_cols.size(); ++i) {
        translated_cols_[i] = local_cols[i] + col_offset_;
    }

    global_view_.addMatrixEntries(translated_rows_, translated_cols_, values);
}

void BlockView::addVectorEntries(
    std::span<const GlobalIndex> local_indices,
    std::span<const Real> values)
{
    translated_rows_.resize(local_indices.size());

    for (std::size_t i = 0; i < local_indices.size(); ++i) {
        translated_rows_[i] = local_indices[i] + row_offset_;
    }

    global_view_.addVectorEntries(translated_rows_, values);
}

// ============================================================================
// Two-Field Block Kernel Implementation
// ============================================================================

/**
 * @brief Block kernel for 2-field systems (e.g., Stokes)
 */
class TwoFieldBlockKernel : public IBlockKernel {
public:
    TwoFieldBlockKernel(AssemblyKernel& a_kernel,
                        AssemblyKernel& b_kernel,
                        AssemblyKernel* bt_kernel,
                        AssemblyKernel* c_kernel)
        : a_kernel_(a_kernel)
        , b_kernel_(b_kernel)
        , bt_kernel_(bt_kernel)
        , c_kernel_(c_kernel)
    {
    }

    void computeBlock(
        AssemblyContext& context,
        FieldId row_field,
        FieldId col_field,
        KernelOutput& output) override
    {
        if (row_field == 0 && col_field == 0) {
            // A block (velocity-velocity)
            a_kernel_.computeCell(context, output);
        } else if (row_field == 0 && col_field == 1) {
            // B block (velocity-pressure)
            b_kernel_.computeCell(context, output);
        } else if (row_field == 1 && col_field == 0) {
            // B^T block (pressure-velocity)
            if (bt_kernel_) {
                bt_kernel_->computeCell(context, output);
            } else {
                // Assume B^T is transpose of B
                // Would need to compute B and transpose
                output.clear();
            }
        } else if (row_field == 1 && col_field == 1) {
            // C block (pressure-pressure)
            if (c_kernel_) {
                c_kernel_->computeCell(context, output);
            } else {
                output.clear();  // Zero block
            }
        }
    }

    void computeRhs(
        AssemblyContext& context,
        FieldId field,
        KernelOutput& output) override
    {
        if (field == 0) {
            // Velocity RHS
            a_kernel_.computeCell(context, output);
        } else {
            // Pressure RHS (usually zero for Stokes)
            output.clear();
        }
    }

    [[nodiscard]] bool hasBlock(FieldId row_field, FieldId col_field) const override {
        if (row_field == 0 && col_field == 0) return true;   // A
        if (row_field == 0 && col_field == 1) return true;   // B
        if (row_field == 1 && col_field == 0) return true;   // B^T
        if (row_field == 1 && col_field == 1) return c_kernel_ != nullptr;  // C
        return false;
    }

    [[nodiscard]] int numFields() const override { return 2; }

private:
    AssemblyKernel& a_kernel_;
    AssemblyKernel& b_kernel_;
    AssemblyKernel* bt_kernel_;
    AssemblyKernel* c_kernel_;
};

// ============================================================================
// BlockAssembler Implementation
// ============================================================================

BlockAssembler::BlockAssembler()
    : options_{}
    , loop_(std::make_unique<AssemblyLoop>())
{
}

BlockAssembler::BlockAssembler(const BlockAssemblerOptions& options)
    : options_(options)
    , loop_(std::make_unique<AssemblyLoop>())
{
}

BlockAssembler::~BlockAssembler() = default;

BlockAssembler::BlockAssembler(BlockAssembler&& other) noexcept = default;

BlockAssembler& BlockAssembler::operator=(BlockAssembler&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void BlockAssembler::setMesh(const IMeshAccess& mesh) {
    mesh_ = &mesh;
    loop_->setMesh(mesh);
}

void BlockAssembler::addField(
    FieldId id,
    const std::string& name,
    const spaces::FunctionSpace& space,
    const dofs::DofMap& dof_map)
{
    FieldConfig field;
    field.id = id;
    field.name = name;
    field.space = &space;
    field.dof_map = &dof_map;
    field.constraints = nullptr;
    field.components = 1;
    field.is_pressure_like = false;

    config_.fields.push_back(field);

    // Recompute block offsets
    computeBlockOffsets();
}

void BlockAssembler::setFieldConstraints(
    FieldId id,
    const constraints::AffineConstraints& constraints)
{
    for (auto& field : config_.fields) {
        if (field.id == id) {
            field.constraints = &constraints;
            return;
        }
    }
    FE_THROW(FEException, "Field not found: " + std::to_string(id));
}

void BlockAssembler::setBlockDofMap(const dofs::BlockDofMap& block_dof_map) {
    block_dof_map_ = &block_dof_map;
}

void BlockAssembler::setKernel(IBlockKernel& kernel) {
    kernel_ = &kernel;
}

void BlockAssembler::setOptions(const BlockAssemblerOptions& options) {
    options_ = options;
}

bool BlockAssembler::isConfigured() const noexcept {
    return mesh_ != nullptr &&
           !config_.fields.empty() &&
           kernel_ != nullptr;
}

void BlockAssembler::computeBlockOffsets() {
    int n_fields = numFields();
    field_offsets_.resize(static_cast<std::size_t>(n_fields + 1));
    field_sizes_.resize(static_cast<std::size_t>(n_fields));

    field_offsets_[0] = 0;
    for (int i = 0; i < n_fields; ++i) {
        auto idx = static_cast<std::size_t>(i);
        // Get field size from DOF map
        // This is a placeholder - actual implementation would query DofMap
        GlobalIndex field_size = 100;  // Placeholder
        field_sizes_[idx] = field_size;
        field_offsets_[idx + 1] = field_offsets_[idx] + field_size;
    }
}

std::pair<GlobalIndex, GlobalIndex> BlockAssembler::getBlockOffset(
    FieldId row_field,
    FieldId col_field) const
{
    auto row_idx = static_cast<std::size_t>(row_field);
    auto col_idx = static_cast<std::size_t>(col_field);

    FE_THROW_IF(row_idx >= field_offsets_.size() - 1, "Invalid row field");
    FE_THROW_IF(col_idx >= field_offsets_.size() - 1, "Invalid col field");

    return {field_offsets_[row_idx], field_offsets_[col_idx]};
}

std::pair<GlobalIndex, GlobalIndex> BlockAssembler::getBlockSize(
    FieldId row_field,
    FieldId col_field) const
{
    auto row_idx = static_cast<std::size_t>(row_field);
    auto col_idx = static_cast<std::size_t>(col_field);

    FE_THROW_IF(row_idx >= field_sizes_.size(), "Invalid row field");
    FE_THROW_IF(col_idx >= field_sizes_.size(), "Invalid col field");

    return {field_sizes_[row_idx], field_sizes_[col_idx]};
}

GlobalIndex BlockAssembler::totalSize() const noexcept {
    if (field_offsets_.empty()) return 0;
    return field_offsets_.back();
}

// ============================================================================
// Monolithic Assembly
// ============================================================================

BlockAssemblyStats BlockAssembler::assembleSystem(
    GlobalSystemView& matrix_view,
    GlobalSystemView& rhs_view)
{
    FE_THROW_IF(!isConfigured(), "BlockAssembler not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = BlockAssemblyStats{};

    // Begin assembly
    matrix_view.beginAssemblyPhase();
    rhs_view.beginAssemblyPhase();

    int n_fields = numFields();

    // Assemble all blocks
    for (int i = 0; i < n_fields; ++i) {
        for (int j = 0; j < n_fields; ++j) {
            FieldId row_field = static_cast<FieldId>(i);
            FieldId col_field = static_cast<FieldId>(j);

            if (kernel_->hasBlock(row_field, col_field)) {
                auto [row_offset, col_offset] = getBlockOffset(row_field, col_field);
                assembleBlockInternal(row_field, col_field, matrix_view,
                                     row_offset, col_offset);
            }
        }

        // Assemble field RHS
        FieldId field = static_cast<FieldId>(i);
        GlobalIndex rhs_offset = field_offsets_[static_cast<std::size_t>(i)];
        assembleFieldRhsInternal(field, rhs_view, rhs_offset);
    }

    // Finalize assembly
    matrix_view.endAssemblyPhase();
    rhs_view.endAssemblyPhase();

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

BlockAssemblyStats BlockAssembler::assembleMatrix(GlobalSystemView& matrix_view) {
    FE_THROW_IF(!isConfigured(), "BlockAssembler not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = BlockAssemblyStats{};

    matrix_view.beginAssemblyPhase();

    int n_fields = numFields();

    for (int i = 0; i < n_fields; ++i) {
        for (int j = 0; j < n_fields; ++j) {
            FieldId row_field = static_cast<FieldId>(i);
            FieldId col_field = static_cast<FieldId>(j);

            if (kernel_->hasBlock(row_field, col_field)) {
                auto [row_offset, col_offset] = getBlockOffset(row_field, col_field);
                assembleBlockInternal(row_field, col_field, matrix_view,
                                     row_offset, col_offset);
            }
        }
    }

    matrix_view.endAssemblyPhase();

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

BlockAssemblyStats BlockAssembler::assembleRhs(GlobalSystemView& rhs_view) {
    FE_THROW_IF(!isConfigured(), "BlockAssembler not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = BlockAssemblyStats{};

    rhs_view.beginAssemblyPhase();

    int n_fields = numFields();

    for (int i = 0; i < n_fields; ++i) {
        FieldId field = static_cast<FieldId>(i);
        GlobalIndex rhs_offset = field_offsets_[static_cast<std::size_t>(i)];
        assembleFieldRhsInternal(field, rhs_view, rhs_offset);
    }

    rhs_view.endAssemblyPhase();

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

// ============================================================================
// Block-wise Assembly
// ============================================================================

BlockAssemblyStats BlockAssembler::assembleBlock(
    FieldId row_field,
    FieldId col_field,
    GlobalSystemView& block_view)
{
    FE_THROW_IF(!isConfigured(), "BlockAssembler not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = BlockAssemblyStats{};

    block_view.beginAssemblyPhase();

    // For separate block assembly, offset is 0 (block_view is already the block)
    assembleBlockInternal(row_field, col_field, block_view, 0, 0);

    block_view.endAssemblyPhase();

    auto end_time = std::chrono::high_resolution_clock::now();

    BlockIndex block_idx(row_field, col_field);
    last_stats_.block_assembly_seconds[block_idx] =
        std::chrono::duration<double>(end_time - start_time).count();
    last_stats_.total_seconds = last_stats_.block_assembly_seconds[block_idx];

    return last_stats_;
}

BlockAssemblyStats BlockAssembler::assembleFieldRhs(
    FieldId field,
    GlobalSystemView& rhs_view)
{
    FE_THROW_IF(!isConfigured(), "BlockAssembler not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = BlockAssemblyStats{};

    rhs_view.beginAssemblyPhase();

    // For separate field RHS, offset is 0
    assembleFieldRhsInternal(field, rhs_view, 0);

    rhs_view.endAssemblyPhase();

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

// ============================================================================
// Coupling Assembly
// ============================================================================

BlockAssemblyStats BlockAssembler::assembleCouplingBlocks(GlobalSystemView& matrix_view) {
    FE_THROW_IF(!isConfigured(), "BlockAssembler not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = BlockAssemblyStats{};

    matrix_view.beginAssemblyPhase();

    int n_fields = numFields();

    for (int i = 0; i < n_fields; ++i) {
        for (int j = 0; j < n_fields; ++j) {
            if (i == j) continue;  // Skip diagonal blocks

            FieldId row_field = static_cast<FieldId>(i);
            FieldId col_field = static_cast<FieldId>(j);

            if (kernel_->hasBlock(row_field, col_field)) {
                auto [row_offset, col_offset] = getBlockOffset(row_field, col_field);
                assembleBlockInternal(row_field, col_field, matrix_view,
                                     row_offset, col_offset);
            }
        }
    }

    matrix_view.endAssemblyPhase();

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

BlockAssemblyStats BlockAssembler::assembleDiagonalBlocks(GlobalSystemView& matrix_view) {
    FE_THROW_IF(!isConfigured(), "BlockAssembler not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = BlockAssemblyStats{};

    matrix_view.beginAssemblyPhase();

    int n_fields = numFields();

    for (int i = 0; i < n_fields; ++i) {
        FieldId field = static_cast<FieldId>(i);

        if (kernel_->hasBlock(field, field)) {
            auto [row_offset, col_offset] = getBlockOffset(field, field);
            assembleBlockInternal(field, field, matrix_view, row_offset, col_offset);
        }
    }

    matrix_view.endAssemblyPhase();

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

// ============================================================================
// Internal Assembly
// ============================================================================

void BlockAssembler::assembleBlockInternal(
    FieldId row_field,
    FieldId col_field,
    GlobalSystemView& matrix_view,
    GlobalIndex row_offset,
    GlobalIndex col_offset)
{
    // Initialize thread-local storage if needed
    int num_threads = options_.num_threads;
    if (num_threads <= 0) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = 1;
#endif
    }

    if (thread_contexts_.size() < static_cast<std::size_t>(num_threads)) {
        thread_contexts_.resize(static_cast<std::size_t>(num_threads));
        thread_outputs_.resize(static_cast<std::size_t>(num_threads));
        thread_row_dofs_.resize(static_cast<std::size_t>(num_threads));
        thread_col_dofs_.resize(static_cast<std::size_t>(num_threads));

        for (int i = 0; i < num_threads; ++i) {
            thread_contexts_[static_cast<std::size_t>(i)] =
                std::make_unique<AssemblyContext>();
        }
    }

    // Cell loop for this block
    mesh_->forEachCell([this, row_field, col_field, &matrix_view,
                        row_offset, col_offset](GlobalIndex cell_id) {
        // Thread ID for sequential case
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif

        auto& context = *thread_contexts_[static_cast<std::size_t>(tid)];
        auto& output = thread_outputs_[static_cast<std::size_t>(tid)];
        auto& row_dofs = thread_row_dofs_[static_cast<std::size_t>(tid)];
        auto& col_dofs = thread_col_dofs_[static_cast<std::size_t>(tid)];

        // Get DOFs for this cell and fields
        getCellFieldDofs(cell_id, row_field, row_dofs);
        getCellFieldDofs(cell_id, col_field, col_dofs);

        // Skip if no DOFs
        if (row_dofs.empty() || col_dofs.empty()) return;

        // Prepare context
        // context.prepare(cell_id, ...);

        // Compute block
        output.clear();
        output.reserve(static_cast<LocalIndex>(row_dofs.size()),
                       static_cast<LocalIndex>(col_dofs.size()), true, false);

        kernel_->computeBlock(context, row_field, col_field, output);

        // Translate DOFs to global and insert
        for (auto& dof : row_dofs) {
            dof += row_offset;
        }
        for (auto& dof : col_dofs) {
            dof += col_offset;
        }

        matrix_view.addMatrixEntries(row_dofs, col_dofs, output.local_matrix);
    });

    ++last_stats_.num_cells;
}

void BlockAssembler::assembleFieldRhsInternal(
    FieldId field,
    GlobalSystemView& rhs_view,
    GlobalIndex offset)
{
    // Similar to assembleBlockInternal but for RHS

    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif

    if (thread_contexts_.empty()) {
        thread_contexts_.push_back(std::make_unique<AssemblyContext>());
        thread_outputs_.resize(1);
        thread_row_dofs_.resize(1);
    }

    auto& context = *thread_contexts_[static_cast<std::size_t>(tid)];
    auto& output = thread_outputs_[static_cast<std::size_t>(tid)];
    auto& dofs = thread_row_dofs_[static_cast<std::size_t>(tid)];

    mesh_->forEachCell([this, field, &rhs_view, offset,
                        &context, &output, &dofs](GlobalIndex cell_id) {
        getCellFieldDofs(cell_id, field, dofs);

        if (dofs.empty()) return;

        // Compute RHS
        output.clear();
        output.reserve(static_cast<LocalIndex>(dofs.size()),
                       static_cast<LocalIndex>(dofs.size()), false, true);

        kernel_->computeRhs(context, field, output);

        // Translate and insert
        for (auto& dof : dofs) {
            dof += offset;
        }

        rhs_view.addVectorEntries(dofs, output.local_vector);
    });
}

void BlockAssembler::getCellFieldDofs(
    GlobalIndex /*cell_id*/,
    FieldId /*field*/,
    std::vector<GlobalIndex>& dofs)
{
    // This would query the field's DOF map for DOFs on this cell
    // Placeholder implementation

    dofs.clear();

    // In actual implementation:
    // const auto& field_config = *config_.getField(field);
    // field_config.dof_map->getCellDofs(cell_id, dofs);
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<BlockAssembler> createBlockAssembler(
    const BlockAssemblerOptions& options)
{
    return std::make_unique<BlockAssembler>(options);
}

std::unique_ptr<IBlockKernel> createTwoFieldBlockKernel(
    AssemblyKernel& a_kernel,
    AssemblyKernel& b_kernel,
    AssemblyKernel* bt_kernel,
    AssemblyKernel* c_kernel)
{
    return std::make_unique<TwoFieldBlockKernel>(
        a_kernel, b_kernel, bt_kernel, c_kernel);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
