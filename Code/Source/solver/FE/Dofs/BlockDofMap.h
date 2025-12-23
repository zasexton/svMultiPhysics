/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_DOFS_BLOCKDOFMAP_H
#define SVMP_FE_DOFS_BLOCKDOFMAP_H

/**
 * @file BlockDofMap.h
 * @brief Block-structured DOF management for multi-physics
 *
 * BlockDofMap extends FieldDofMap with explicit block structure support,
 * designed for:
 *  - Block preconditioners (Schur complement, block-Jacobi)
 *  - Segregated solvers (solve each physics separately)
 *  - Mixed finite elements (velocity-pressure coupling)
 *  - Monolithic vs split formulations
 *
 * The block structure can be:
 *  - Monolithic: All fields in one system, with block structure for preconditioning
 *  - Segregated: Each block solved independently
 *  - Hybrid: Some blocks coupled, others decoupled
 */

#include "FieldDofMap.h"
#include "DofMap.h"
#include "DofIndexSet.h"
#include "SubspaceView.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <string>
#include <memory>
#include <optional>

namespace svmp {
namespace FE {
namespace dofs {

/**
 * @brief Block coupling type
 */
enum class BlockCoupling : std::uint8_t {
    None,       ///< Blocks are decoupled
    OneWay,     ///< One block depends on another
    TwoWay,     ///< Blocks are mutually coupled
    Full        ///< Full coupling (all entries nonzero)
};

/**
 * @brief Coupling matrix for block pairs
 */
struct BlockCouplingInfo {
    std::size_t block_i;
    std::size_t block_j;
    BlockCoupling coupling;
};

/**
 * @brief Block solver strategy
 */
enum class BlockSolverStrategy : std::uint8_t {
    Monolithic,     ///< Solve all blocks simultaneously
    Segregated,     ///< Solve each block separately
    GaussSeidel,    ///< Block Gauss-Seidel iteration
    SchurComplement ///< Use Schur complement for saddle-point
};

/**
 * @brief Block-structured DOF management
 *
 * Provides block-level operations on top of FieldDofMap:
 * - Extract block submatrices and subvectors
 * - Define coupling between blocks
 * - Support different solver strategies
 */
class BlockDofMap {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    BlockDofMap();
    ~BlockDofMap();

    // Move semantics
    BlockDofMap(BlockDofMap&&) noexcept;
    BlockDofMap& operator=(BlockDofMap&&) noexcept;

    // No copy
    BlockDofMap(const BlockDofMap&) = delete;
    BlockDofMap& operator=(const BlockDofMap&) = delete;

    // =========================================================================
    // Block Definition
    // =========================================================================

    /**
     * @brief Add a block with given name and DOF count
     *
     * @param name Block name
     * @param n_dofs Number of DOFs in this block
     * @return Block index
     */
    int addBlock(const std::string& name, GlobalIndex n_dofs);

    /**
     * @brief Add a block from field DOF map
     *
     * @param name Block name
     * @param field_map Field DOF map containing this block's DOFs
     * @param field_name Name of field in field_map
     * @return Block index
     */
    int addBlock(const std::string& name, const FieldDofMap& field_map,
                 const std::string& field_name);

    /**
     * @brief Add a block from explicit DOF range
     *
     * @param name Block name
     * @param start_dof First DOF in block
     * @param end_dof One past last DOF in block
     * @return Block index
     */
    int addBlock(const std::string& name, GlobalIndex start_dof, GlobalIndex end_dof);

    /**
     * @brief Set coupling between blocks
     *
     * @param block_i First block index
     * @param block_j Second block index
     * @param coupling Coupling type
     */
    void setCoupling(std::size_t block_i, std::size_t block_j, BlockCoupling coupling);

    /**
     * @brief Set coupling between blocks by name
     */
    void setCoupling(const std::string& name_i, const std::string& name_j,
                     BlockCoupling coupling);

    /**
     * @brief Finalize block structure
     */
    void finalize();

    // =========================================================================
    // Block Queries
    // =========================================================================

    /**
     * @brief Get number of blocks
     */
    [[nodiscard]] std::size_t numBlocks() const noexcept { return blocks_.size(); }

    /**
     * @brief Get block index by name
     * @return Block index, or -1 if not found
     */
    [[nodiscard]] int getBlockIndex(const std::string& name) const;

    /**
     * @brief Get block name
     */
    [[nodiscard]] const std::string& getBlockName(std::size_t block_idx) const;

    /**
     * @brief Get block DOF range
     * @return (start, end) DOF indices
     */
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getBlockRange(std::size_t block_idx) const;

    /**
     * @brief Get block size (number of DOFs)
     */
    [[nodiscard]] GlobalIndex getBlockSize(std::size_t block_idx) const;

    /**
     * @brief Get all block sizes
     */
    [[nodiscard]] std::vector<GlobalIndex> getBlockSizes() const;

    /**
     * @brief Get all block offsets
     */
    [[nodiscard]] std::vector<GlobalIndex> getBlockOffsets() const;

    /**
     * @brief Get total number of DOFs
     */
    [[nodiscard]] GlobalIndex totalDofs() const noexcept { return total_dofs_; }

    // =========================================================================
    // Coupling Queries
    // =========================================================================

    /**
     * @brief Get coupling between blocks
     */
    [[nodiscard]] BlockCoupling getCoupling(std::size_t block_i, std::size_t block_j) const;

    /**
     * @brief Check if blocks are coupled
     */
    [[nodiscard]] bool areCoupled(std::size_t block_i, std::size_t block_j) const;

    /**
     * @brief Get all coupling information
     */
    [[nodiscard]] std::vector<BlockCouplingInfo> getAllCouplings() const;

    /**
     * @brief Check if system has saddle-point structure
     *
     * True if there's a block (typically pressure) with no self-coupling
     * that is coupled to another block.
     */
    [[nodiscard]] bool hasSaddlePointStructure() const;

    // =========================================================================
    // SubspaceView Generation
    // =========================================================================

    /**
     * @brief Get subspace view for a block
     */
    [[nodiscard]] std::unique_ptr<SubspaceView> getBlockView(std::size_t block_idx) const;

    /**
     * @brief Get subspace view for a block by name
     */
    [[nodiscard]] std::unique_ptr<SubspaceView> getBlockView(const std::string& name) const;

    /**
     * @brief Get subspace view for multiple blocks
     */
    [[nodiscard]] std::unique_ptr<SubspaceView> getBlocksView(
        std::span<const std::size_t> block_indices) const;

    // =========================================================================
    // Block Matrix/Vector Operations
    // =========================================================================

    /**
     * @brief Extract block from matrix (dense representation)
     *
     * @param matrix Full system matrix (row-major)
     * @param block_i Row block
     * @param block_j Column block
     * @return Block submatrix (row-major)
     */
    [[nodiscard]] std::vector<double> extractBlock(
        std::span<const double> matrix,
        std::size_t block_i, std::size_t block_j) const;

    /**
     * @brief Extract block from vector
     *
     * @param vector Full system vector
     * @param block_idx Block index
     * @return Block subvector
     */
    [[nodiscard]] std::vector<double> extractBlockVector(
        std::span<const double> vector,
        std::size_t block_idx) const;

    /**
     * @brief Scatter block to full matrix
     *
     * @param block_matrix Block submatrix
     * @param block_i Row block
     * @param block_j Column block
     * @param full_matrix Full system matrix (modified)
     */
    void scatterBlock(std::span<const double> block_matrix,
                      std::size_t block_i, std::size_t block_j,
                      std::span<double> full_matrix) const;

    /**
     * @brief Scatter block to full vector
     */
    void scatterBlockVector(std::span<const double> block_vector,
                            std::size_t block_idx,
                            std::span<double> full_vector) const;

    // =========================================================================
    // DOF Mapping
    // =========================================================================

    /**
     * @brief Map block-local DOF to global DOF
     *
     * @param block_idx Block index
     * @param local_dof DOF index within block
     * @return Global DOF index
     */
    [[nodiscard]] GlobalIndex blockToGlobal(std::size_t block_idx, GlobalIndex local_dof) const;

    /**
     * @brief Map global DOF to block-local DOF
     *
     * @param global_dof Global DOF index
     * @return (block_idx, local_dof), or nullopt if invalid
     */
    [[nodiscard]] std::optional<std::pair<std::size_t, GlobalIndex>>
    globalToBlock(GlobalIndex global_dof) const;

    // =========================================================================
    // State
    // =========================================================================

    [[nodiscard]] bool isFinalized() const noexcept { return finalized_; }

private:
    // Block descriptor
    struct BlockInfo {
        std::string name;
        GlobalIndex start_dof{0};
        GlobalIndex end_dof{0};
        GlobalIndex size{0};
    };

    // Block storage
    std::vector<BlockInfo> blocks_;
    std::unordered_map<std::string, std::size_t> name_to_index_;

    // Coupling matrix (stored as flat array for n_blocks x n_blocks)
    std::vector<BlockCoupling> coupling_matrix_;

    // Totals
    GlobalIndex total_dofs_{0};

    // State
    bool finalized_{false};

    // Helpers
    void checkFinalized() const;
    void checkNotFinalized() const;
    void checkBlockIndex(std::size_t idx) const;
};

} // namespace dofs
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_DOFS_BLOCKDOFMAP_H
