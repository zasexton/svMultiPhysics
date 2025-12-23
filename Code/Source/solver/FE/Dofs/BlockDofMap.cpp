/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "BlockDofMap.h"
#include <algorithm>
#include <numeric>

namespace svmp {
namespace FE {
namespace dofs {

// =============================================================================
// Construction
// =============================================================================

BlockDofMap::BlockDofMap() = default;
BlockDofMap::~BlockDofMap() = default;

BlockDofMap::BlockDofMap(BlockDofMap&&) noexcept = default;
BlockDofMap& BlockDofMap::operator=(BlockDofMap&&) noexcept = default;

// =============================================================================
// Block Definition
// =============================================================================

int BlockDofMap::addBlock(const std::string& name, GlobalIndex n_dofs) {
    checkNotFinalized();

    if (name_to_index_.count(name) > 0) {
        throw FEException("BlockDofMap::addBlock: block '" + name + "' already exists");
    }

    BlockInfo info;
    info.name = name;
    info.start_dof = total_dofs_;
    info.end_dof = total_dofs_ + n_dofs;
    info.size = n_dofs;

    auto idx = static_cast<int>(blocks_.size());
    blocks_.push_back(std::move(info));
    name_to_index_[name] = static_cast<std::size_t>(idx);

    total_dofs_ += n_dofs;

    return idx;
}

int BlockDofMap::addBlock(const std::string& name, const FieldDofMap& field_map,
                          const std::string& field_name) {
    checkNotFinalized();

    auto range = field_map.getFieldDofRange(field_name);
    GlobalIndex n_dofs = range.second - range.first;

    return addBlock(name, n_dofs);
}

int BlockDofMap::addBlock(const std::string& name, GlobalIndex start_dof, GlobalIndex end_dof) {
    checkNotFinalized();

    if (end_dof <= start_dof) {
        throw FEException("BlockDofMap::addBlock: invalid DOF range");
    }

    if (name_to_index_.count(name) > 0) {
        throw FEException("BlockDofMap::addBlock: block '" + name + "' already exists");
    }

    BlockInfo info;
    info.name = name;
    info.start_dof = start_dof;
    info.end_dof = end_dof;
    info.size = end_dof - start_dof;

    auto idx = static_cast<int>(blocks_.size());
    blocks_.push_back(std::move(info));
    name_to_index_[name] = static_cast<std::size_t>(idx);

    total_dofs_ = std::max(total_dofs_, end_dof);

    return idx;
}

void BlockDofMap::setCoupling(std::size_t block_i, std::size_t block_j, BlockCoupling coupling) {
    checkNotFinalized();

    auto n = blocks_.size();
    if (block_i >= n || block_j >= n) {
        throw FEException("BlockDofMap::setCoupling: invalid block index");
    }

    // Resize coupling matrix if needed
    if (coupling_matrix_.size() != n * n) {
        coupling_matrix_.resize(n * n, BlockCoupling::None);
    }

    coupling_matrix_[block_i * n + block_j] = coupling;
}

void BlockDofMap::setCoupling(const std::string& name_i, const std::string& name_j,
                               BlockCoupling coupling) {
    auto idx_i = getBlockIndex(name_i);
    auto idx_j = getBlockIndex(name_j);

    if (idx_i < 0 || idx_j < 0) {
        throw FEException("BlockDofMap::setCoupling: block not found");
    }

    setCoupling(static_cast<std::size_t>(idx_i), static_cast<std::size_t>(idx_j), coupling);
}

void BlockDofMap::finalize() {
    checkNotFinalized();

    // Initialize coupling matrix if not set
    auto n = blocks_.size();
    if (coupling_matrix_.size() != n * n) {
        coupling_matrix_.resize(n * n, BlockCoupling::Full);  // Default: all coupled
    }

    finalized_ = true;
}

// =============================================================================
// Block Queries
// =============================================================================

int BlockDofMap::getBlockIndex(const std::string& name) const {
    auto it = name_to_index_.find(name);
    if (it == name_to_index_.end()) {
        return -1;
    }
    return static_cast<int>(it->second);
}

const std::string& BlockDofMap::getBlockName(std::size_t block_idx) const {
    checkBlockIndex(block_idx);
    return blocks_[block_idx].name;
}

std::pair<GlobalIndex, GlobalIndex> BlockDofMap::getBlockRange(std::size_t block_idx) const {
    checkBlockIndex(block_idx);
    const auto& block = blocks_[block_idx];
    return {block.start_dof, block.end_dof};
}

GlobalIndex BlockDofMap::getBlockSize(std::size_t block_idx) const {
    checkBlockIndex(block_idx);
    return blocks_[block_idx].size;
}

std::vector<GlobalIndex> BlockDofMap::getBlockSizes() const {
    std::vector<GlobalIndex> sizes;
    sizes.reserve(blocks_.size());
    for (const auto& block : blocks_) {
        sizes.push_back(block.size);
    }
    return sizes;
}

std::vector<GlobalIndex> BlockDofMap::getBlockOffsets() const {
    std::vector<GlobalIndex> offsets;
    offsets.reserve(blocks_.size());
    for (const auto& block : blocks_) {
        offsets.push_back(block.start_dof);
    }
    return offsets;
}

// =============================================================================
// Coupling Queries
// =============================================================================

BlockCoupling BlockDofMap::getCoupling(std::size_t block_i, std::size_t block_j) const {
    auto n = blocks_.size();
    if (block_i >= n || block_j >= n) {
        return BlockCoupling::None;
    }

    if (coupling_matrix_.size() != n * n) {
        return BlockCoupling::Full;  // Default
    }

    return coupling_matrix_[block_i * n + block_j];
}

bool BlockDofMap::areCoupled(std::size_t block_i, std::size_t block_j) const {
    return getCoupling(block_i, block_j) != BlockCoupling::None;
}

std::vector<BlockCouplingInfo> BlockDofMap::getAllCouplings() const {
    std::vector<BlockCouplingInfo> couplings;

    auto n = blocks_.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            auto coupling = getCoupling(i, j);
            if (coupling != BlockCoupling::None) {
                couplings.push_back({i, j, coupling});
            }
        }
    }

    return couplings;
}

bool BlockDofMap::hasSaddlePointStructure() const {
    // Check for a block that has no self-coupling but is coupled to others
    auto n = blocks_.size();
    if (n < 2) return false;

    for (std::size_t i = 0; i < n; ++i) {
        if (getCoupling(i, i) == BlockCoupling::None) {
            // Check if coupled to any other block
            for (std::size_t j = 0; j < n; ++j) {
                if (i != j && areCoupled(i, j)) {
                    return true;
                }
            }
        }
    }

    return false;
}

// =============================================================================
// SubspaceView Generation
// =============================================================================

std::unique_ptr<SubspaceView> BlockDofMap::getBlockView(std::size_t block_idx) const {
    checkBlockIndex(block_idx);

    const auto& block = blocks_[block_idx];

    return std::make_unique<SubspaceView>(
        block.start_dof, block.end_dof,
        block.name,
        static_cast<int>(block_idx));
}

std::unique_ptr<SubspaceView> BlockDofMap::getBlockView(const std::string& name) const {
    auto idx = getBlockIndex(name);
    if (idx < 0) {
        throw FEException("BlockDofMap::getBlockView: block '" + name + "' not found");
    }
    return getBlockView(static_cast<std::size_t>(idx));
}

std::unique_ptr<SubspaceView> BlockDofMap::getBlocksView(
    std::span<const std::size_t> block_indices) const {

    std::vector<GlobalIndex> all_dofs;

    for (auto idx : block_indices) {
        checkBlockIndex(idx);
        const auto& block = blocks_[idx];
        for (GlobalIndex d = block.start_dof; d < block.end_dof; ++d) {
            all_dofs.push_back(d);
        }
    }

    return std::make_unique<SubspaceView>(IndexSet(std::move(all_dofs)), "combined_blocks");
}

// =============================================================================
// Block Matrix/Vector Operations
// =============================================================================

std::vector<double> BlockDofMap::extractBlock(
    std::span<const double> matrix,
    std::size_t block_i, std::size_t block_j) const {

    checkBlockIndex(block_i);
    checkBlockIndex(block_j);

    const auto& row_block = blocks_[block_i];
    const auto& col_block = blocks_[block_j];

    auto n_rows = row_block.size;
    auto n_cols = col_block.size;

    std::vector<double> result(static_cast<std::size_t>(n_rows * n_cols));

    // Assuming matrix is stored row-major with total_dofs columns
    for (GlobalIndex i = 0; i < n_rows; ++i) {
        GlobalIndex global_row = row_block.start_dof + i;
        for (GlobalIndex j = 0; j < n_cols; ++j) {
            GlobalIndex global_col = col_block.start_dof + j;
            auto mat_idx = static_cast<std::size_t>(global_row * total_dofs_ + global_col);
            auto res_idx = static_cast<std::size_t>(i * n_cols + j);

            if (mat_idx < matrix.size()) {
                result[res_idx] = matrix[mat_idx];
            }
        }
    }

    return result;
}

std::vector<double> BlockDofMap::extractBlockVector(
    std::span<const double> vector,
    std::size_t block_idx) const {

    checkBlockIndex(block_idx);

    const auto& block = blocks_[block_idx];
    std::vector<double> result(static_cast<std::size_t>(block.size));

    for (GlobalIndex i = 0; i < block.size; ++i) {
        auto global_idx = static_cast<std::size_t>(block.start_dof + i);
        if (global_idx < vector.size()) {
            result[static_cast<std::size_t>(i)] = vector[global_idx];
        }
    }

    return result;
}

void BlockDofMap::scatterBlock(std::span<const double> block_matrix,
                                std::size_t block_i, std::size_t block_j,
                                std::span<double> full_matrix) const {

    checkBlockIndex(block_i);
    checkBlockIndex(block_j);

    const auto& row_block = blocks_[block_i];
    const auto& col_block = blocks_[block_j];

    auto n_rows = row_block.size;
    auto n_cols = col_block.size;

    for (GlobalIndex i = 0; i < n_rows; ++i) {
        GlobalIndex global_row = row_block.start_dof + i;
        for (GlobalIndex j = 0; j < n_cols; ++j) {
            GlobalIndex global_col = col_block.start_dof + j;
            auto mat_idx = static_cast<std::size_t>(global_row * total_dofs_ + global_col);
            auto blk_idx = static_cast<std::size_t>(i * n_cols + j);

            if (mat_idx < full_matrix.size() && blk_idx < block_matrix.size()) {
                full_matrix[mat_idx] = block_matrix[blk_idx];
            }
        }
    }
}

void BlockDofMap::scatterBlockVector(std::span<const double> block_vector,
                                      std::size_t block_idx,
                                      std::span<double> full_vector) const {

    checkBlockIndex(block_idx);

    const auto& block = blocks_[block_idx];

    for (GlobalIndex i = 0; i < block.size; ++i) {
        auto global_idx = static_cast<std::size_t>(block.start_dof + i);
        auto local_idx = static_cast<std::size_t>(i);

        if (global_idx < full_vector.size() && local_idx < block_vector.size()) {
            full_vector[global_idx] = block_vector[local_idx];
        }
    }
}

// =============================================================================
// DOF Mapping
// =============================================================================

GlobalIndex BlockDofMap::blockToGlobal(std::size_t block_idx, GlobalIndex local_dof) const {
    checkBlockIndex(block_idx);

    const auto& block = blocks_[block_idx];
    if (local_dof < 0 || local_dof >= block.size) {
        throw FEException("BlockDofMap::blockToGlobal: local DOF out of range");
    }

    return block.start_dof + local_dof;
}

std::optional<std::pair<std::size_t, GlobalIndex>>
BlockDofMap::globalToBlock(GlobalIndex global_dof) const {

    if (global_dof < 0 || global_dof >= total_dofs_) {
        return std::nullopt;
    }

    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const auto& block = blocks_[b];
        if (global_dof >= block.start_dof && global_dof < block.end_dof) {
            return std::make_pair(b, global_dof - block.start_dof);
        }
    }

    return std::nullopt;
}

// =============================================================================
// Helpers
// =============================================================================

void BlockDofMap::checkFinalized() const {
    if (!finalized_) {
        throw FEException("BlockDofMap: operation requires finalization");
    }
}

void BlockDofMap::checkNotFinalized() const {
    if (finalized_) {
        throw FEException("BlockDofMap: operation not allowed after finalization");
    }
}

void BlockDofMap::checkBlockIndex(std::size_t idx) const {
    if (idx >= blocks_.size()) {
        throw FEException("BlockDofMap: invalid block index");
    }
}

} // namespace dofs
} // namespace FE
} // namespace svmp
