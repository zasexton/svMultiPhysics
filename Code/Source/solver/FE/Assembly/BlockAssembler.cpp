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
#include "Dofs/DofMap.h"
#include "StandardAssembler.h"

#include <chrono>
#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <optional>

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

namespace {

[[nodiscard]] bool kernelNeedsSolution(const AssemblyKernel& kernel)
{
    const auto required = kernel.getRequiredData();
    return hasFlag(required, RequiredData::SolutionCoefficients) ||
           hasFlag(required, RequiredData::SolutionValues) ||
           hasFlag(required, RequiredData::SolutionGradients) ||
           hasFlag(required, RequiredData::SolutionHessians) ||
           hasFlag(required, RequiredData::SolutionLaplacians);
}

[[nodiscard]] std::string blockLabel(FieldId row, FieldId col)
{
    return "(" + std::to_string(row) + "," + std::to_string(col) + ")";
}

/**
 * @brief GlobalSystemView adapter that shifts indices before forwarding.
 *
 * This is used to support:
 * - Monolithic assembly when an assembler does not support DOF offsets (shift +offset)
 * - Block-wise assembly with global-indexed assemblers (shift -offset)
 */
class ShiftedSystemView final : public GlobalSystemView {
public:
    ShiftedSystemView(GlobalSystemView& base, GlobalIndex row_shift, GlobalIndex col_shift)
        : base_(base)
        , row_shift_(row_shift)
        , col_shift_(col_shift)
    {
    }

    void addMatrixEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> local_matrix,
        AddMode mode = AddMode::Add) override
    {
        FE_THROW_IF(row_shift_ != col_shift_, FEException,
                    "ShiftedSystemView::addMatrixEntries(square): row/col shifts differ");

        shifted_dofs_.resize(dofs.size());
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            shifted_dofs_[i] = dofs[i] + row_shift_;
        }
        base_.addMatrixEntries(shifted_dofs_, local_matrix, mode);
    }

    void addMatrixEntries(
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        std::span<const Real> local_matrix,
        AddMode mode = AddMode::Add) override
    {
        shifted_rows_.resize(row_dofs.size());
        shifted_cols_.resize(col_dofs.size());

        for (std::size_t i = 0; i < row_dofs.size(); ++i) {
            shifted_rows_[i] = row_dofs[i] + row_shift_;
        }
        for (std::size_t i = 0; i < col_dofs.size(); ++i) {
            shifted_cols_[i] = col_dofs[i] + col_shift_;
        }

        base_.addMatrixEntries(shifted_rows_, shifted_cols_, local_matrix, mode);
    }

    void addMatrixEntry(GlobalIndex row, GlobalIndex col, Real value, AddMode mode = AddMode::Add) override
    {
        base_.addMatrixEntry(row + row_shift_, col + col_shift_, value, mode);
    }

    void setDiagonal(std::span<const GlobalIndex> dofs, std::span<const Real> values) override
    {
        FE_THROW_IF(row_shift_ != col_shift_, FEException,
                    "ShiftedSystemView::setDiagonal: row/col shifts differ");

        shifted_dofs_.resize(dofs.size());
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            shifted_dofs_[i] = dofs[i] + row_shift_;
        }
        base_.setDiagonal(shifted_dofs_, values);
    }

    void setDiagonal(GlobalIndex dof, Real value) override
    {
        FE_THROW_IF(row_shift_ != col_shift_, FEException,
                    "ShiftedSystemView::setDiagonal(single): row/col shifts differ");
        base_.setDiagonal(dof + row_shift_, value);
    }

    void zeroRows(std::span<const GlobalIndex> rows, bool set_diagonal = true) override
    {
        shifted_rows_.resize(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            shifted_rows_[i] = rows[i] + row_shift_;
        }
        base_.zeroRows(shifted_rows_, set_diagonal);
    }

    void addVectorEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> local_vector,
        AddMode mode = AddMode::Add) override
    {
        shifted_dofs_.resize(dofs.size());
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            shifted_dofs_[i] = dofs[i] + row_shift_;
        }
        base_.addVectorEntries(shifted_dofs_, local_vector, mode);
    }

    void addVectorEntry(GlobalIndex dof, Real value, AddMode mode = AddMode::Add) override
    {
        base_.addVectorEntry(dof + row_shift_, value, mode);
    }

    void setVectorEntries(std::span<const GlobalIndex> dofs, std::span<const Real> values) override
    {
        shifted_dofs_.resize(dofs.size());
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            shifted_dofs_[i] = dofs[i] + row_shift_;
        }
        base_.setVectorEntries(shifted_dofs_, values);
    }

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override
    {
        shifted_dofs_.resize(dofs.size());
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            shifted_dofs_[i] = dofs[i] + row_shift_;
        }
        base_.zeroVectorEntries(shifted_dofs_);
    }

    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override
    {
        return base_.getVectorEntry(dof + row_shift_);
    }

    void beginAssemblyPhase() override { base_.beginAssemblyPhase(); }
    void endAssemblyPhase() override { base_.endAssemblyPhase(); }
    void finalizeAssembly() override { base_.finalizeAssembly(); }
    [[nodiscard]] AssemblyPhase getPhase() const noexcept override { return base_.getPhase(); }

    [[nodiscard]] bool hasMatrix() const noexcept override { return base_.hasMatrix(); }
    [[nodiscard]] bool hasVector() const noexcept override { return base_.hasVector(); }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return base_.numRows(); }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return base_.numCols(); }
    [[nodiscard]] std::string backendName() const override { return base_.backendName(); }
    void zero() override { base_.zero(); }

    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override
    {
        return base_.getMatrixEntry(row + row_shift_, col + col_shift_);
    }

private:
    GlobalSystemView& base_;
    GlobalIndex row_shift_{0};
    GlobalIndex col_shift_{0};

    mutable std::vector<GlobalIndex> shifted_dofs_;
    mutable std::vector<GlobalIndex> shifted_rows_;
    mutable std::vector<GlobalIndex> shifted_cols_;
};

/**
 * @brief GlobalSystemView adapter that suppresses begin/end/finalize lifecycle calls.
 *
 * Useful for orchestrators that want to manage begin/end/finalize externally while
 * still calling assemblers that unconditionally invoke these methods.
 */
class NoFinalizeSystemView final : public GlobalSystemView {
public:
    explicit NoFinalizeSystemView(GlobalSystemView& base)
        : base_(base)
    {
    }

    void addMatrixEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> local_matrix,
        AddMode mode = AddMode::Add) override
    {
        base_.addMatrixEntries(dofs, local_matrix, mode);
    }

    void addMatrixEntries(
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        std::span<const Real> local_matrix,
        AddMode mode = AddMode::Add) override
    {
        base_.addMatrixEntries(row_dofs, col_dofs, local_matrix, mode);
    }

    void addMatrixEntry(GlobalIndex row, GlobalIndex col, Real value, AddMode mode = AddMode::Add) override
    {
        base_.addMatrixEntry(row, col, value, mode);
    }

    void setDiagonal(std::span<const GlobalIndex> dofs, std::span<const Real> values) override
    {
        base_.setDiagonal(dofs, values);
    }

    void setDiagonal(GlobalIndex dof, Real value) override { base_.setDiagonal(dof, value); }

    void zeroRows(std::span<const GlobalIndex> rows, bool set_diagonal = true) override
    {
        base_.zeroRows(rows, set_diagonal);
    }

    void addVectorEntries(
        std::span<const GlobalIndex> dofs,
        std::span<const Real> local_vector,
        AddMode mode = AddMode::Add) override
    {
        base_.addVectorEntries(dofs, local_vector, mode);
    }

    void addVectorEntry(GlobalIndex dof, Real value, AddMode mode = AddMode::Add) override
    {
        base_.addVectorEntry(dof, value, mode);
    }

    void setVectorEntries(std::span<const GlobalIndex> dofs, std::span<const Real> values) override
    {
        base_.setVectorEntries(dofs, values);
    }

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override { base_.zeroVectorEntries(dofs); }

    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override { return base_.getVectorEntry(dof); }

    void beginAssemblyPhase() override {}
    void endAssemblyPhase() override {}
    void finalizeAssembly() override {}
    [[nodiscard]] AssemblyPhase getPhase() const noexcept override { return base_.getPhase(); }

    [[nodiscard]] bool hasMatrix() const noexcept override { return base_.hasMatrix(); }
    [[nodiscard]] bool hasVector() const noexcept override { return base_.hasVector(); }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return base_.numRows(); }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return base_.numCols(); }
    [[nodiscard]] std::string backendName() const override { return base_.backendName(); }
    void zero() override { base_.zero(); }

    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override
    {
        return base_.getMatrixEntry(row, col);
    }

private:
    GlobalSystemView& base_;
};

} // namespace

// ============================================================================
// BlockAssembler Implementation
// ============================================================================

BlockAssembler::BlockAssembler()
    : options_{}
{
}

BlockAssembler::BlockAssembler(const BlockAssemblerOptions& options)
    : options_(options)
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

void BlockAssembler::setOptions(const BlockAssemblerOptions& options) {
    options_ = options;
}

bool BlockAssembler::isConfigured() const noexcept {
    return mesh_ != nullptr && !config_.fields.empty();
}

void BlockAssembler::computeBlockOffsets() {
    if (config_.fields.empty()) {
        field_offsets_.clear();
        field_sizes_.clear();
        return;
    }

    FieldId max_id = 0;
    for (const auto& f : config_.fields) {
        max_id = std::max(max_id, f.id);
    }

    const auto n_slots = static_cast<std::size_t>(max_id) + 1;
    field_sizes_.assign(n_slots, 0);
    field_offsets_.assign(n_slots + 1, 0);

    std::vector<bool> seen(n_slots, false);
    for (const auto& f : config_.fields) {
        const auto idx = static_cast<std::size_t>(f.id);
        FE_THROW_IF(idx >= n_slots, FEException,
                    "BlockAssembler::computeBlockOffsets: field id out of range");
        FE_THROW_IF(seen[idx], FEException,
                    "BlockAssembler::computeBlockOffsets: duplicate field id " + std::to_string(f.id));
        seen[idx] = true;

        FE_CHECK_NOT_NULL(f.dof_map, "BlockAssembler::computeBlockOffsets: field dof_map");
        field_sizes_[idx] = f.dof_map->getNumDofs();
    }

    for (std::size_t i = 0; i < n_slots; ++i) {
        field_offsets_[i + 1] = field_offsets_[i] + field_sizes_[i];
    }
}

std::pair<GlobalIndex, GlobalIndex> BlockAssembler::getBlockOffset(
    FieldId row_field,
    FieldId col_field) const
{
    FE_THROW_IF(config_.getField(row_field) == nullptr, FEException,
                "BlockAssembler::getBlockOffset: row field not found: " + std::to_string(row_field));
    FE_THROW_IF(config_.getField(col_field) == nullptr, FEException,
                "BlockAssembler::getBlockOffset: col field not found: " + std::to_string(col_field));

    const auto row_idx = static_cast<std::size_t>(row_field);
    const auto col_idx = static_cast<std::size_t>(col_field);

    FE_THROW_IF(field_offsets_.empty(), FEException,
                "BlockAssembler::getBlockOffset: offsets not computed");
    FE_THROW_IF(row_idx >= field_offsets_.size() - 1, FEException, "Invalid row field id");
    FE_THROW_IF(col_idx >= field_offsets_.size() - 1, FEException, "Invalid col field id");

    return {field_offsets_[row_idx], field_offsets_[col_idx]};
}

std::pair<GlobalIndex, GlobalIndex> BlockAssembler::getBlockSize(
    FieldId row_field,
    FieldId col_field) const
{
    FE_THROW_IF(config_.getField(row_field) == nullptr, FEException,
                "BlockAssembler::getBlockSize: row field not found: " + std::to_string(row_field));
    FE_THROW_IF(config_.getField(col_field) == nullptr, FEException,
                "BlockAssembler::getBlockSize: col field not found: " + std::to_string(col_field));

    const auto row_idx = static_cast<std::size_t>(row_field);
    const auto col_idx = static_cast<std::size_t>(col_field);

    FE_THROW_IF(field_sizes_.empty(), FEException,
                "BlockAssembler::getBlockSize: sizes not computed");
    FE_THROW_IF(row_idx >= field_sizes_.size(), FEException, "Invalid row field id");
    FE_THROW_IF(col_idx >= field_sizes_.size(), FEException, "Invalid col field id");

    return {field_sizes_[row_idx], field_sizes_[col_idx]};
}

GlobalIndex BlockAssembler::totalSize() const noexcept {
    if (field_offsets_.empty()) return 0;
    return field_offsets_.back();
}

// ============================================================================
// Assembler Assignment (per-block)
// ============================================================================

void BlockAssembler::setBlockAssembler(
    FieldId row_field,
    FieldId col_field,
    std::shared_ptr<Assembler> assembler)
{
    FE_THROW_IF(config_.getField(row_field) == nullptr, FEException,
                "BlockAssembler::setBlockAssembler: row field not found: " + std::to_string(row_field));
    FE_THROW_IF(config_.getField(col_field) == nullptr, FEException,
                "BlockAssembler::setBlockAssembler: col field not found: " + std::to_string(col_field));

    if (!assembler) {
        block_assemblers_.erase({row_field, col_field});
        return;
    }

    // Propagate cached state
    if (!current_solution_.empty()) {
        assembler->setCurrentSolution(current_solution_);
    }
    assembler->setTime(time_);
    assembler->setTimeStep(dt_);

    block_assemblers_[{row_field, col_field}] = std::move(assembler);
}

void BlockAssembler::setFieldAssembler(FieldId field, std::shared_ptr<Assembler> assembler)
{
    FE_THROW_IF(config_.getField(field) == nullptr, FEException,
                "BlockAssembler::setFieldAssembler: field not found: " + std::to_string(field));

    for (const auto& f : config_.fields) {
        setBlockAssembler(field, f.id, assembler);
        setBlockAssembler(f.id, field, assembler);
    }
}

void BlockAssembler::setDiagonalAssembler(FieldId field, std::shared_ptr<Assembler> assembler)
{
    FE_THROW_IF(config_.getField(field) == nullptr, FEException,
                "BlockAssembler::setDiagonalAssembler: field not found: " + std::to_string(field));
    setBlockAssembler(field, field, std::move(assembler));
}

void BlockAssembler::setDefaultAssembler(std::shared_ptr<Assembler> assembler)
{
    default_assembler_ = std::move(assembler);
    if (default_assembler_) {
        if (!current_solution_.empty()) {
            default_assembler_->setCurrentSolution(current_solution_);
        }
        default_assembler_->setTime(time_);
        default_assembler_->setTimeStep(dt_);
    }
}

Assembler& BlockAssembler::getBlockAssembler(FieldId row_field, FieldId col_field)
{
    auto it = block_assemblers_.find({row_field, col_field});
    if (it != block_assemblers_.end() && it->second) {
        return *it->second;
    }

    if (!default_assembler_) {
        default_assembler_ = std::make_shared<StandardAssembler>();
        if (!current_solution_.empty()) {
            default_assembler_->setCurrentSolution(current_solution_);
        }
        default_assembler_->setTime(time_);
        default_assembler_->setTimeStep(dt_);
    }

    return *default_assembler_;
}

bool BlockAssembler::hasBlockAssembler(FieldId row_field, FieldId col_field) const
{
    auto it = block_assemblers_.find({row_field, col_field});
    return it != block_assemblers_.end() && it->second != nullptr;
}

// ============================================================================
// Kernel Assignment (per-block / per-field)
// ============================================================================

void BlockAssembler::setBlockKernel(
    FieldId row_field,
    FieldId col_field,
    std::shared_ptr<AssemblyKernel> kernel)
{
    FE_THROW_IF(config_.getField(row_field) == nullptr, FEException,
                "BlockAssembler::setBlockKernel: row field not found: " + std::to_string(row_field));
    FE_THROW_IF(config_.getField(col_field) == nullptr, FEException,
                "BlockAssembler::setBlockKernel: col field not found: " + std::to_string(col_field));

    const BlockIndex idx{row_field, col_field};
    if (!kernel) {
        block_kernels_.erase(idx);
        return;
    }

    block_kernels_[idx] = std::move(kernel);
}

void BlockAssembler::setRhsKernel(FieldId field, std::shared_ptr<AssemblyKernel> kernel)
{
    FE_THROW_IF(config_.getField(field) == nullptr, FEException,
                "BlockAssembler::setRhsKernel: field not found: " + std::to_string(field));

    if (!kernel) {
        rhs_kernels_.erase(field);
        return;
    }

    rhs_kernels_[field] = std::move(kernel);
}

bool BlockAssembler::hasBlockKernel(FieldId row_field, FieldId col_field) const
{
    return block_kernels_.find({row_field, col_field}) != block_kernels_.end();
}

std::vector<BlockIndex> BlockAssembler::getNonZeroBlocks() const
{
    std::vector<BlockIndex> blocks;
    blocks.reserve(block_kernels_.size());
    for (const auto& [idx, kernel] : block_kernels_) {
        if (kernel) {
            blocks.push_back(idx);
        }
    }
    return blocks;
}

// ============================================================================
// State Propagation
// ============================================================================

void BlockAssembler::setCurrentSolution(std::span<const Real> global_solution)
{
    current_solution_ = global_solution;

    for (auto& [_, assembler] : block_assemblers_) {
        if (assembler) {
            assembler->setCurrentSolution(global_solution);
        }
    }
    if (default_assembler_) {
        default_assembler_->setCurrentSolution(global_solution);
    }
}

void BlockAssembler::setTime(Real time)
{
    time_ = time;
    for (auto& [_, assembler] : block_assemblers_) {
        if (assembler) {
            assembler->setTime(time_);
        }
    }
    if (default_assembler_) {
        default_assembler_->setTime(time_);
    }
}

void BlockAssembler::setTimeStep(Real dt)
{
    dt_ = dt;
    for (auto& [_, assembler] : block_assemblers_) {
        if (assembler) {
            assembler->setTimeStep(dt_);
        }
    }
    if (default_assembler_) {
        default_assembler_->setTimeStep(dt_);
    }
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
    last_stats_.num_cells = mesh_->numCells();

    for (const auto& [idx, kernel] : block_kernels_) {
        if (!kernel) {
            continue;
        }
        assembleBlockInternal(idx.row_field, idx.col_field, matrix_view, true);
    }

    for (const auto& [field, kernel] : rhs_kernels_) {
        if (!kernel) {
            continue;
        }
        assembleFieldRhsInternal(field, rhs_view, true);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

BlockAssemblyStats BlockAssembler::assembleMatrix(GlobalSystemView& matrix_view) {
    FE_THROW_IF(!isConfigured(), "BlockAssembler not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = BlockAssemblyStats{};
    last_stats_.num_cells = mesh_->numCells();

    for (const auto& [idx, kernel] : block_kernels_) {
        if (!kernel) {
            continue;
        }
        assembleBlockInternal(idx.row_field, idx.col_field, matrix_view, true);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

BlockAssemblyStats BlockAssembler::assembleRhs(GlobalSystemView& rhs_view) {
    FE_THROW_IF(!isConfigured(), "BlockAssembler not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = BlockAssemblyStats{};
    last_stats_.num_cells = mesh_->numCells();

    for (const auto& [field, kernel] : rhs_kernels_) {
        if (!kernel) {
            continue;
        }
        assembleFieldRhsInternal(field, rhs_view, true);
    }

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
    last_stats_.num_cells = mesh_->numCells();

    assembleBlockInternal(row_field, col_field, block_view, false);

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
    last_stats_.num_cells = mesh_->numCells();

    assembleFieldRhsInternal(field, rhs_view, false);

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

// ============================================================================
// Parallel Block Assembly
// ============================================================================

BlockAssemblyStats BlockAssembler::assembleBlocksParallel(
    std::span<const BlockIndex> blocks,
    GlobalSystemView& matrix_view,
    GlobalSystemView* rhs_view)
{
    FE_THROW_IF(!isConfigured(), "BlockAssembler not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = BlockAssemblyStats{};
    last_stats_.num_cells = mesh_->numCells();

    FE_THROW_IF(field_offsets_.size() < 2 || field_sizes_.empty(), FEException,
                "BlockAssembler::assembleBlocksParallel: field offsets not computed");

    // Build global field access registry once (used by offset-capable assemblers).
    std::vector<FieldSolutionAccess> field_access;
    field_access.reserve(config_.fields.size());
    for (const auto& f : config_.fields) {
        if (!f.space || !f.dof_map) {
            continue;
        }
        const auto f_offset = getBlockOffset(f.id, f.id).first;
        field_access.push_back(FieldSolutionAccess{
            .field = f.id,
            .space = f.space,
            .dof_map = f.dof_map,
            .dof_offset = f_offset,
        });
    }

    // Locks to prevent concurrent use of shared field DOFs / shared assembler instances.
    const std::size_t n_field_slots = field_offsets_.size() - 1;
    std::vector<std::mutex> field_mutexes(n_field_slots);

    std::map<const Assembler*, std::size_t> assembler_mutex_index;
    std::vector<std::unique_ptr<std::mutex>> assembler_mutexes;
    auto ensureAssemblerMutex = [&](const Assembler* ptr) -> std::size_t {
        auto it = assembler_mutex_index.find(ptr);
        if (it != assembler_mutex_index.end()) {
            return it->second;
        }
        const std::size_t idx = assembler_mutexes.size();
        assembler_mutexes.push_back(std::make_unique<std::mutex>());
        assembler_mutex_index.emplace(ptr, idx);
        return idx;
    };

    struct MatrixJob {
        BlockIndex idx{};
        const FieldConfig* row_config{nullptr};
        const FieldConfig* col_config{nullptr};
        AssemblyKernel* kernel{nullptr};
        Assembler* assembler{nullptr};
        GlobalIndex row_offset{0};
        GlobalIndex col_offset{0};
        bool supports_offsets{false};
        std::size_t assembler_mutex{0};
    };

    std::vector<MatrixJob> jobs;
    jobs.reserve(blocks.size());

    for (const auto& b : blocks) {
        auto kit = block_kernels_.find(b);
        if (kit == block_kernels_.end() || !kit->second) {
            continue;
        }

        const FieldConfig* row_config = config_.getField(b.row_field);
        const FieldConfig* col_config = config_.getField(b.col_field);
        FE_THROW_IF(row_config == nullptr, FEException,
                    "BlockAssembler::assembleBlocksParallel: row field not found: " + std::to_string(b.row_field));
        FE_THROW_IF(col_config == nullptr, FEException,
                    "BlockAssembler::assembleBlocksParallel: col field not found: " + std::to_string(b.col_field));
        FE_CHECK_NOT_NULL(row_config->space, "BlockAssembler::assembleBlocksParallel: row space");
        FE_CHECK_NOT_NULL(col_config->space, "BlockAssembler::assembleBlocksParallel: col space");
        FE_CHECK_NOT_NULL(row_config->dof_map, "BlockAssembler::assembleBlocksParallel: row dof map");
        FE_CHECK_NOT_NULL(col_config->dof_map, "BlockAssembler::assembleBlocksParallel: col dof map");

        Assembler& assembler = getBlockAssembler(b.row_field, b.col_field);
        const bool supports_offsets = assembler.supportsDofOffsets();

        const auto [row_offset, col_offset] = getBlockOffset(b.row_field, b.col_field);

        jobs.push_back(MatrixJob{
            .idx = b,
            .row_config = row_config,
            .col_config = col_config,
            .kernel = kit->second.get(),
            .assembler = &assembler,
            .row_offset = row_offset,
            .col_offset = col_offset,
            .supports_offsets = supports_offsets,
            .assembler_mutex = ensureAssemblerMutex(&assembler),
        });
    }

    // Begin assembly once; per-job finalize is suppressed via NoFinalizeSystemView.
    matrix_view.beginAssemblyPhase();
    const bool rhs_is_distinct = (rhs_view != nullptr && rhs_view != &matrix_view);
    if (rhs_is_distinct) {
        rhs_view->beginAssemblyPhase();
    } else if (rhs_view != nullptr && rhs_view == &matrix_view) {
        // Nothing to do (already begun).
    }

    int num_threads = options_.num_threads;
    if (num_threads <= 0) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = 1;
#endif
    }

    std::vector<AssemblyResult> results(jobs.size());
    std::exception_ptr first_error;
    std::mutex error_mutex;
    std::atomic<bool> abort{false};

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
#endif
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        if (abort.load(std::memory_order_relaxed)) {
            continue;
        }

        try {
            const auto& job = jobs[i];

            const auto row_idx = static_cast<std::size_t>(job.idx.row_field);
            const auto col_idx = static_cast<std::size_t>(job.idx.col_field);
            FE_THROW_IF(row_idx >= field_mutexes.size() || col_idx >= field_mutexes.size(), FEException,
                        "BlockAssembler::assembleBlocksParallel: field id out of range");

            auto& asm_mutex = *assembler_mutexes[job.assembler_mutex];

            // Lock the involved fields (DOF-disjoint criterion) and the assembler instance.
            if (row_idx == col_idx) {
                std::scoped_lock lock(field_mutexes[row_idx], asm_mutex);

                NoFinalizeSystemView nofinal(matrix_view);
                std::optional<ShiftedSystemView> shifted;
                GlobalSystemView* view = &nofinal;

                const GlobalIndex row_shift = job.supports_offsets ? 0 : job.row_offset;
                const GlobalIndex col_shift = job.supports_offsets ? 0 : job.col_offset;
                if (row_shift != 0 || col_shift != 0) {
                    shifted.emplace(nofinal, row_shift, col_shift);
                    view = &*shifted;
                }

                if (job.supports_offsets) {
                    job.assembler->setRowDofMap(*job.row_config->dof_map, job.row_offset);
                    job.assembler->setColDofMap(*job.col_config->dof_map, job.col_offset);
                    job.assembler->setFieldSolutionAccess(field_access);
                } else {
                    job.assembler->setRowDofMap(*job.row_config->dof_map, 0);
                    job.assembler->setColDofMap(*job.col_config->dof_map, 0);
                }

                const constraints::AffineConstraints* constraints =
                    (options_.apply_constraints ? job.row_config->constraints : nullptr);
                job.assembler->setConstraints(constraints);
                job.assembler->setTime(time_);
                job.assembler->setTimeStep(dt_);

                if (kernelNeedsSolution(*job.kernel)) {
                    FE_THROW_IF(!job.assembler->supportsSolution(), FEException,
                                "BlockAssembler::assembleBlocksParallel: kernel requires solution but assembler '" +
                                    job.assembler->name() + "' does not support solution");
                    FE_THROW_IF(current_solution_.empty(), FEException,
                                "BlockAssembler::assembleBlocksParallel: kernel requires solution but no solution was set");
                }

                const auto field_reqs = job.kernel->fieldRequirements();
                if (!field_reqs.empty()) {
                    FE_THROW_IF(!job.assembler->supportsFieldRequirements(), FEException,
                                "BlockAssembler::assembleBlocksParallel: kernel declares fieldRequirements() but assembler '" +
                                    job.assembler->name() + "' does not support field requirements");
                }

                if (!current_solution_.empty()) {
                    if (job.supports_offsets) {
                        job.assembler->setCurrentSolution(current_solution_);
                    } else {
                        FE_THROW_IF(!field_reqs.empty(), FEException,
                                    "BlockAssembler::assembleBlocksParallel: assembler '" + job.assembler->name() +
                                        "' does not support DOF offsets; cannot satisfy kernel fieldRequirements() for block " +
                                        blockLabel(job.idx.row_field, job.idx.col_field));

                        const auto trial = static_cast<std::size_t>(job.idx.col_field);
                        const auto off = static_cast<std::size_t>(field_offsets_[trial]);
                        const auto sz = static_cast<std::size_t>(field_sizes_[trial]);
                        FE_THROW_IF(off + sz > current_solution_.size(), FEException,
                                    "BlockAssembler::assembleBlocksParallel: solution vector too small for field slice");
                        job.assembler->setCurrentSolution(current_solution_.subspan(off, sz));
                    }
                }

                job.assembler->initialize();
                results[i] = job.assembler->assembleMatrix(*mesh_, *job.row_config->space, *job.col_config->space,
                                                           *job.kernel, *view);
                job.assembler->finalize(view, nullptr);
            } else {
                std::scoped_lock lock(field_mutexes[row_idx], field_mutexes[col_idx], asm_mutex);

                NoFinalizeSystemView nofinal(matrix_view);
                std::optional<ShiftedSystemView> shifted;
                GlobalSystemView* view = &nofinal;

                const GlobalIndex row_shift = job.supports_offsets ? 0 : job.row_offset;
                const GlobalIndex col_shift = job.supports_offsets ? 0 : job.col_offset;
                if (row_shift != 0 || col_shift != 0) {
                    shifted.emplace(nofinal, row_shift, col_shift);
                    view = &*shifted;
                }

                if (job.supports_offsets) {
                    job.assembler->setRowDofMap(*job.row_config->dof_map, job.row_offset);
                    job.assembler->setColDofMap(*job.col_config->dof_map, job.col_offset);
                    job.assembler->setFieldSolutionAccess(field_access);
                } else {
                    job.assembler->setRowDofMap(*job.row_config->dof_map, 0);
                    job.assembler->setColDofMap(*job.col_config->dof_map, 0);
                }

                const constraints::AffineConstraints* constraints =
                    (options_.apply_constraints ? job.row_config->constraints : nullptr);
                job.assembler->setConstraints(constraints);
                job.assembler->setTime(time_);
                job.assembler->setTimeStep(dt_);

                if (kernelNeedsSolution(*job.kernel)) {
                    FE_THROW_IF(!job.assembler->supportsSolution(), FEException,
                                "BlockAssembler::assembleBlocksParallel: kernel requires solution but assembler '" +
                                    job.assembler->name() + "' does not support solution");
                    FE_THROW_IF(current_solution_.empty(), FEException,
                                "BlockAssembler::assembleBlocksParallel: kernel requires solution but no solution was set");
                }

                const auto field_reqs = job.kernel->fieldRequirements();
                if (!field_reqs.empty()) {
                    FE_THROW_IF(!job.assembler->supportsFieldRequirements(), FEException,
                                "BlockAssembler::assembleBlocksParallel: kernel declares fieldRequirements() but assembler '" +
                                    job.assembler->name() + "' does not support field requirements");
                }

                if (!current_solution_.empty()) {
                    if (job.supports_offsets) {
                        job.assembler->setCurrentSolution(current_solution_);
                    } else {
                        FE_THROW_IF(!field_reqs.empty(), FEException,
                                    "BlockAssembler::assembleBlocksParallel: assembler '" + job.assembler->name() +
                                        "' does not support DOF offsets; cannot satisfy kernel fieldRequirements() for block " +
                                        blockLabel(job.idx.row_field, job.idx.col_field));

                        const auto trial = static_cast<std::size_t>(job.idx.col_field);
                        const auto off = static_cast<std::size_t>(field_offsets_[trial]);
                        const auto sz = static_cast<std::size_t>(field_sizes_[trial]);
                        FE_THROW_IF(off + sz > current_solution_.size(), FEException,
                                    "BlockAssembler::assembleBlocksParallel: solution vector too small for field slice");
                        job.assembler->setCurrentSolution(current_solution_.subspan(off, sz));
                    }
                }

                job.assembler->initialize();
                results[i] = job.assembler->assembleMatrix(*mesh_, *job.row_config->space, *job.col_config->space,
                                                           *job.kernel, *view);
                job.assembler->finalize(view, nullptr);
            }
        } catch (...) {
            abort.store(true, std::memory_order_relaxed);
            std::scoped_lock lock(error_mutex);
            if (!first_error) {
                first_error = std::current_exception();
            }
        }
    }

    if (first_error) {
        std::rethrow_exception(first_error);
    }

    for (std::size_t i = 0; i < jobs.size(); ++i) {
        last_stats_.block_assembly_seconds[jobs[i].idx] = results[i].elapsed_time_seconds;
        last_stats_.block_nnz[jobs[i].idx] = results[i].matrix_entries_inserted;
    }

    // Optionally assemble RHS for fields involved in the requested blocks (row fields only).
    if (rhs_view != nullptr) {
        struct RhsJob {
            FieldId field{0};
            const FieldConfig* config{nullptr};
            AssemblyKernel* kernel{nullptr};
            Assembler* assembler{nullptr};
            GlobalIndex offset{0};
            bool supports_offsets{false};
            std::size_t assembler_mutex{0};
        };

        std::set<FieldId> fields_to_assemble;
        for (const auto& b : blocks) {
            fields_to_assemble.insert(b.row_field);
        }

        std::vector<RhsJob> rhs_jobs;
        rhs_jobs.reserve(fields_to_assemble.size());
        for (FieldId f : fields_to_assemble) {
            auto it = rhs_kernels_.find(f);
            if (it == rhs_kernels_.end() || !it->second) {
                continue;
            }

            const FieldConfig* cfg = config_.getField(f);
            FE_THROW_IF(cfg == nullptr, FEException,
                        "BlockAssembler::assembleBlocksParallel(RHS): field not found: " + std::to_string(f));
            FE_CHECK_NOT_NULL(cfg->space, "BlockAssembler::assembleBlocksParallel(RHS): space");
            FE_CHECK_NOT_NULL(cfg->dof_map, "BlockAssembler::assembleBlocksParallel(RHS): dof map");

            Assembler& assembler = getBlockAssembler(f, f);
            const bool supports_offsets = assembler.supportsDofOffsets();
            const auto [offset, _] = getBlockOffset(f, f);

            rhs_jobs.push_back(RhsJob{
                .field = f,
                .config = cfg,
                .kernel = it->second.get(),
                .assembler = &assembler,
                .offset = offset,
                .supports_offsets = supports_offsets,
                .assembler_mutex = ensureAssemblerMutex(&assembler),
            });
        }

        // Reset error handling for the RHS pass.
        abort.store(false, std::memory_order_relaxed);
        first_error = nullptr;

        std::vector<AssemblyResult> rhs_results(rhs_jobs.size());

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
#endif
        for (std::size_t i = 0; i < rhs_jobs.size(); ++i) {
            if (abort.load(std::memory_order_relaxed)) {
                continue;
            }

            try {
                const auto& job = rhs_jobs[i];
                const auto field_idx = static_cast<std::size_t>(job.field);
                FE_THROW_IF(field_idx >= field_mutexes.size(), FEException,
                            "BlockAssembler::assembleBlocksParallel(RHS): field id out of range");

                auto& asm_mutex = *assembler_mutexes[job.assembler_mutex];
                std::scoped_lock lock(field_mutexes[field_idx], asm_mutex);

                NoFinalizeSystemView nofinal(*rhs_view);
                std::optional<ShiftedSystemView> shifted;
                GlobalSystemView* view = &nofinal;

                const GlobalIndex shift = job.supports_offsets ? 0 : job.offset;
                if (shift != 0) {
                    shifted.emplace(nofinal, shift, shift);
                    view = &*shifted;
                }

                if (job.supports_offsets) {
                    job.assembler->setRowDofMap(*job.config->dof_map, job.offset);
                    job.assembler->setColDofMap(*job.config->dof_map, job.offset);
                    job.assembler->setFieldSolutionAccess(field_access);
                } else {
                    job.assembler->setRowDofMap(*job.config->dof_map, 0);
                    job.assembler->setColDofMap(*job.config->dof_map, 0);
                }

                const constraints::AffineConstraints* constraints =
                    (options_.apply_constraints ? job.config->constraints : nullptr);
                job.assembler->setConstraints(constraints);
                job.assembler->setTime(time_);
                job.assembler->setTimeStep(dt_);

                if (kernelNeedsSolution(*job.kernel)) {
                    FE_THROW_IF(!job.assembler->supportsSolution(), FEException,
                                "BlockAssembler::assembleBlocksParallel(RHS): kernel requires solution but assembler '" +
                                    job.assembler->name() + "' does not support solution");
                    FE_THROW_IF(current_solution_.empty(), FEException,
                                "BlockAssembler::assembleBlocksParallel(RHS): kernel requires solution but no solution was set");
                }

                const auto field_reqs = job.kernel->fieldRequirements();
                if (!field_reqs.empty()) {
                    FE_THROW_IF(!job.assembler->supportsFieldRequirements(), FEException,
                                "BlockAssembler::assembleBlocksParallel(RHS): kernel declares fieldRequirements() but assembler '" +
                                    job.assembler->name() + "' does not support field requirements");
                }

                if (!current_solution_.empty()) {
                    if (job.supports_offsets) {
                        job.assembler->setCurrentSolution(current_solution_);
                    } else {
                        FE_THROW_IF(!field_reqs.empty(), FEException,
                                    "BlockAssembler::assembleBlocksParallel(RHS): assembler '" + job.assembler->name() +
                                        "' does not support DOF offsets; cannot satisfy kernel fieldRequirements() for field " +
                                        std::to_string(job.field));

                        const auto off = static_cast<std::size_t>(field_offsets_[field_idx]);
                        const auto sz = static_cast<std::size_t>(field_sizes_[field_idx]);
                        FE_THROW_IF(off + sz > current_solution_.size(), FEException,
                                    "BlockAssembler::assembleBlocksParallel(RHS): solution vector too small for field slice");
                        job.assembler->setCurrentSolution(current_solution_.subspan(off, sz));
                    }
                }

                job.assembler->initialize();
                rhs_results[i] = job.assembler->assembleVector(*mesh_, *job.config->space, *job.kernel, *view);
                job.assembler->finalize(nullptr, view);
            } catch (...) {
                abort.store(true, std::memory_order_relaxed);
                std::scoped_lock lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
        }

        if (first_error) {
            std::rethrow_exception(first_error);
        }
    }

    // Finalize views once.
    matrix_view.endAssemblyPhase();
    matrix_view.finalizeAssembly();

    if (rhs_is_distinct) {
        rhs_view->endAssemblyPhase();
        rhs_view->finalizeAssembly();
    } else if (rhs_view != nullptr && rhs_view == &matrix_view) {
        // Already finalized above.
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

// ============================================================================
// Selective Assembly
// ============================================================================

BlockAssemblyStats BlockAssembler::assembleBlocksIf(
    std::function<bool(FieldId row, FieldId col)> predicate,
    GlobalSystemView& matrix_view)
{
    FE_THROW_IF(!isConfigured(), "BlockAssembler not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = BlockAssemblyStats{};
    last_stats_.num_cells = mesh_->numCells();

    for (const auto& [idx, kernel] : block_kernels_) {
        if (!kernel) {
            continue;
        }
        if (!predicate(idx.row_field, idx.col_field)) {
            continue;
        }
        assembleBlockInternal(idx.row_field, idx.col_field, matrix_view, true);
    }

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
    last_stats_.num_cells = mesh_->numCells();

    for (const auto& [idx, kernel] : block_kernels_) {
        if (!kernel || idx.isDiagonal()) {
            continue;
        }
        assembleBlockInternal(idx.row_field, idx.col_field, matrix_view, true);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return last_stats_;
}

BlockAssemblyStats BlockAssembler::assembleDiagonalBlocks(GlobalSystemView& matrix_view) {
    FE_THROW_IF(!isConfigured(), "BlockAssembler not configured");

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = BlockAssemblyStats{};
    last_stats_.num_cells = mesh_->numCells();

    for (const auto& [idx, kernel] : block_kernels_) {
        if (!kernel || !idx.isDiagonal()) {
            continue;
        }
        assembleBlockInternal(idx.row_field, idx.col_field, matrix_view, true);
    }

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
    GlobalSystemView& output_view,
    bool is_monolithic)
{
    const FieldConfig* row_config = config_.getField(row_field);
    const FieldConfig* col_config = config_.getField(col_field);
    FE_CHECK_NOT_NULL(row_config, "BlockAssembler::assembleBlockInternal: row field config");
    FE_CHECK_NOT_NULL(col_config, "BlockAssembler::assembleBlockInternal: col field config");
    FE_CHECK_NOT_NULL(mesh_, "BlockAssembler::assembleBlockInternal: mesh");
    FE_CHECK_NOT_NULL(row_config->space, "BlockAssembler::assembleBlockInternal: row space");
    FE_CHECK_NOT_NULL(col_config->space, "BlockAssembler::assembleBlockInternal: col space");
    FE_CHECK_NOT_NULL(row_config->dof_map, "BlockAssembler::assembleBlockInternal: row dof map");
    FE_CHECK_NOT_NULL(col_config->dof_map, "BlockAssembler::assembleBlockInternal: col dof map");

    const BlockIndex idx{row_field, col_field};
    auto kit = block_kernels_.find(idx);
    if (kit == block_kernels_.end() || !kit->second) {
        return; // Zero block
    }

    AssemblyKernel& kernel = *kit->second;

    Assembler& assembler = getBlockAssembler(row_field, col_field);

    const auto offsets = getBlockOffset(row_field, col_field);
    const GlobalIndex row_offset = offsets.first;
    const GlobalIndex col_offset = offsets.second;
    const bool supports_offsets = assembler.supportsDofOffsets();

    // Configure assembler DOF maps
    if (supports_offsets) {
        assembler.setRowDofMap(*row_config->dof_map, row_offset);
        assembler.setColDofMap(*col_config->dof_map, col_offset);
    } else {
        assembler.setRowDofMap(*row_config->dof_map, 0);
        assembler.setColDofMap(*col_config->dof_map, 0);
    }

    // Constraints are field-wise; apply constraints on the test (row) field.
    const constraints::AffineConstraints* constraints =
        (options_.apply_constraints ? row_config->constraints : nullptr);
    assembler.setConstraints(constraints);

    // Time state (optional)
    assembler.setTime(time_);
    assembler.setTimeStep(dt_);

    // Solution state (optional)
    if (kernelNeedsSolution(kernel)) {
        FE_THROW_IF(!assembler.supportsSolution(), FEException,
                    "BlockAssembler::assembleBlockInternal: kernel requires solution but assembler '" +
                        assembler.name() + "' does not support solution");
        FE_THROW_IF(current_solution_.empty(), FEException,
                    "BlockAssembler::assembleBlockInternal: kernel requires solution but no solution was set");
    }

    const auto field_reqs = kernel.fieldRequirements();
    if (!field_reqs.empty()) {
        FE_THROW_IF(!assembler.supportsFieldRequirements(), FEException,
                    "BlockAssembler::assembleBlockInternal: kernel declares fieldRequirements() but assembler '" +
                        assembler.name() + "' does not support field requirements");
    }

    if (!current_solution_.empty()) {
        if (supports_offsets) {
            assembler.setCurrentSolution(current_solution_);

            // Provide accessors for kernels requesting additional fields.
            std::vector<FieldSolutionAccess> access;
            access.reserve(config_.fields.size());
            for (const auto& f : config_.fields) {
                if (!f.space || !f.dof_map) {
                    continue;
                }
                const auto f_offset = getBlockOffset(f.id, f.id).first;
                access.push_back(FieldSolutionAccess{
                    .field = f.id,
                    .space = f.space,
                    .dof_map = f.dof_map,
                    .dof_offset = f_offset,
                });
            }
            assembler.setFieldSolutionAccess(access);
        } else {
            // Fallback: provide only the trial-field slice.
            FE_THROW_IF(!field_reqs.empty(), FEException,
                        "BlockAssembler::assembleBlockInternal: assembler '" + assembler.name() +
                            "' does not support DOF offsets; cannot satisfy kernel fieldRequirements() for block " +
                            blockLabel(row_field, col_field));

            const auto col_idx = static_cast<std::size_t>(col_field);
            FE_THROW_IF(col_idx >= field_offsets_.size() - 1 || col_idx >= field_sizes_.size(), FEException,
                        "BlockAssembler::assembleBlockInternal: invalid col field id");
            const auto off = static_cast<std::size_t>(field_offsets_[col_idx]);
            const auto sz = static_cast<std::size_t>(field_sizes_[col_idx]);
            FE_THROW_IF(off + sz > current_solution_.size(), FEException,
                        "BlockAssembler::assembleBlockInternal: solution vector too small for field slice");

            assembler.setCurrentSolution(current_solution_.subspan(off, sz));
        }
    }

    // Select view shifting based on output mode and assembler capabilities.
    GlobalIndex row_shift = 0;
    GlobalIndex col_shift = 0;
    if (is_monolithic) {
        row_shift = supports_offsets ? 0 : row_offset;
        col_shift = supports_offsets ? 0 : col_offset;
    } else {
        row_shift = supports_offsets ? -row_offset : 0;
        col_shift = supports_offsets ? -col_offset : 0;
    }

    std::optional<ShiftedSystemView> shifted;
    GlobalSystemView* view = &output_view;
    if (row_shift != 0 || col_shift != 0) {
        shifted.emplace(output_view, row_shift, col_shift);
        view = &*shifted;
    }

    assembler.initialize();
    const auto result = assembler.assembleMatrix(*mesh_, *row_config->space, *col_config->space, kernel, *view);
    assembler.finalize(view, nullptr);

    last_stats_.block_assembly_seconds[idx] = result.elapsed_time_seconds;
    last_stats_.block_nnz[idx] = result.matrix_entries_inserted;
}

void BlockAssembler::assembleFieldRhsInternal(
    FieldId field,
    GlobalSystemView& rhs_view,
    bool is_monolithic)
{
    const FieldConfig* field_config = config_.getField(field);
    FE_CHECK_NOT_NULL(field_config, "BlockAssembler::assembleFieldRhsInternal: field config");
    FE_CHECK_NOT_NULL(mesh_, "BlockAssembler::assembleFieldRhsInternal: mesh");
    FE_CHECK_NOT_NULL(field_config->space, "BlockAssembler::assembleFieldRhsInternal: space");
    FE_CHECK_NOT_NULL(field_config->dof_map, "BlockAssembler::assembleFieldRhsInternal: dof map");

    auto kit = rhs_kernels_.find(field);
    if (kit == rhs_kernels_.end() || !kit->second) {
        return;
    }

    AssemblyKernel& kernel = *kit->second;
    Assembler& assembler = getBlockAssembler(field, field);

    const auto offsets = getBlockOffset(field, field);
    const GlobalIndex offset = offsets.first;
    const bool supports_offsets = assembler.supportsDofOffsets();

    if (supports_offsets) {
        assembler.setRowDofMap(*field_config->dof_map, offset);
        assembler.setColDofMap(*field_config->dof_map, offset);
    } else {
        assembler.setRowDofMap(*field_config->dof_map, 0);
        assembler.setColDofMap(*field_config->dof_map, 0);
    }

    const constraints::AffineConstraints* constraints =
        (options_.apply_constraints ? field_config->constraints : nullptr);
    assembler.setConstraints(constraints);

    assembler.setTime(time_);
    assembler.setTimeStep(dt_);

    if (kernelNeedsSolution(kernel)) {
        FE_THROW_IF(!assembler.supportsSolution(), FEException,
                    "BlockAssembler::assembleFieldRhsInternal: kernel requires solution but assembler '" +
                        assembler.name() + "' does not support solution");
        FE_THROW_IF(current_solution_.empty(), FEException,
                    "BlockAssembler::assembleFieldRhsInternal: kernel requires solution but no solution was set");
    }

    const auto field_reqs = kernel.fieldRequirements();
    if (!field_reqs.empty()) {
        FE_THROW_IF(!assembler.supportsFieldRequirements(), FEException,
                    "BlockAssembler::assembleFieldRhsInternal: kernel declares fieldRequirements() but assembler '" +
                        assembler.name() + "' does not support field requirements");
    }

    if (!current_solution_.empty()) {
        if (supports_offsets) {
            assembler.setCurrentSolution(current_solution_);

            std::vector<FieldSolutionAccess> access;
            access.reserve(config_.fields.size());
            for (const auto& f : config_.fields) {
                if (!f.space || !f.dof_map) {
                    continue;
                }
                const auto f_offset = getBlockOffset(f.id, f.id).first;
                access.push_back(FieldSolutionAccess{
                    .field = f.id,
                    .space = f.space,
                    .dof_map = f.dof_map,
                    .dof_offset = f_offset,
                });
            }
            assembler.setFieldSolutionAccess(access);
        } else {
            FE_THROW_IF(!field_reqs.empty(), FEException,
                        "BlockAssembler::assembleFieldRhsInternal: assembler '" + assembler.name() +
                            "' does not support DOF offsets; cannot satisfy kernel fieldRequirements() for field " +
                            std::to_string(field));

            const auto f_idx = static_cast<std::size_t>(field);
            FE_THROW_IF(f_idx >= field_offsets_.size() - 1 || f_idx >= field_sizes_.size(), FEException,
                        "BlockAssembler::assembleFieldRhsInternal: invalid field id");
            const auto off = static_cast<std::size_t>(field_offsets_[f_idx]);
            const auto sz = static_cast<std::size_t>(field_sizes_[f_idx]);
            FE_THROW_IF(off + sz > current_solution_.size(), FEException,
                        "BlockAssembler::assembleFieldRhsInternal: solution vector too small for field slice");

            assembler.setCurrentSolution(current_solution_.subspan(off, sz));
        }
    }

    GlobalIndex row_shift = 0;
    if (is_monolithic) {
        row_shift = supports_offsets ? 0 : offset;
    } else {
        row_shift = supports_offsets ? -offset : 0;
    }

    std::optional<ShiftedSystemView> shifted;
    GlobalSystemView* view = &rhs_view;
    if (row_shift != 0) {
        shifted.emplace(rhs_view, row_shift, row_shift);
        view = &*shifted;
    }

    assembler.initialize();
    const auto result = assembler.assembleVector(*mesh_, *field_config->space, kernel, *view);
    assembler.finalize(nullptr, view);

    (void)result;
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<BlockAssembler> createBlockAssembler(
    const BlockAssemblerOptions& options)
{
    return std::make_unique<BlockAssembler>(options);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
