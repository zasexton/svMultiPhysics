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

#ifndef SVMP_FE_ASSEMBLY_BLOCK_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_BLOCK_ASSEMBLER_H

/**
 * @file BlockAssembler.h
 * @brief Block-structured assembly for multi-field problems
 *
 * BlockAssembler handles assembly of systems with multiple coupled fields,
 * providing generic infrastructure for field registration, block-to-kernel
 * mapping, and coordinated assembly of block-structured matrices/vectors.
 *
 * Block structure:
 * @code
 *   [A_uu  A_up] [u]   [f_u]
 *   [A_pu  A_pp] [p] = [f_p]
 * @endcode
 *
 * Assembly modes:
 * - Monolithic: Assemble full coupled system as single matrix
 * - Block: Assemble individual blocks separately
 * - Segregated: Assemble blocks for segregated (operator splitting) solvers
 *
 * Key features:
 * - Multiple function spaces per field
 * - Block-wise sparsity patterns
 * - Coupling term assembly between fields
 * - Support for different discretizations per field
 * - Integration with block preconditioners
 *
 * Module boundaries:
 * - This module OWNS: block assembly orchestration
 * - This module does NOT OWN: field discretizations, solvers
 *
 * @see AssemblyLoop for iteration infrastructure
 * @see Dofs/BlockDofMap for block DOF handling
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Assembler.h"
#include "AssemblyKernel.h"
#include "GlobalSystemView.h"

#include <vector>
#include <span>
#include <functional>
#include <memory>
#include <string>
#include <map>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
    class BlockDofMap;
}

namespace spaces {
    class FunctionSpace;
}

namespace constraints {
    class AffineConstraints;
}

namespace assembly {

// ============================================================================
// Block Assembly Types
// ============================================================================

/**
 * @brief Field identifier for block systems
 */
using FieldId = std::uint16_t;

/**
 * @brief Block identifier (row_field, col_field)
 */
struct BlockIndex {
    FieldId row_field;
    FieldId col_field;

    BlockIndex() = default;
    BlockIndex(FieldId r, FieldId c) : row_field(r), col_field(c) {}

    bool operator==(const BlockIndex& other) const {
        return row_field == other.row_field && col_field == other.col_field;
    }

    bool operator<(const BlockIndex& other) const {
        if (row_field != other.row_field) return row_field < other.row_field;
        return col_field < other.col_field;
    }

    bool isDiagonal() const { return row_field == col_field; }
};

/**
 * @brief Assembly mode for block systems
 */
enum class BlockAssemblyMode : std::uint8_t {
    Monolithic,     ///< Single coupled matrix
    Block,          ///< Separate block matrices
    Segregated      ///< For operator splitting methods
};

/**
 * @brief Options for block assembly
 */
struct BlockAssemblerOptions {
    /**
     * @brief Assembly mode
     */
    BlockAssemblyMode mode{BlockAssemblyMode::Monolithic};

    /**
     * @brief Number of threads for parallel assembly
     */
    int num_threads{0};

    /**
     * @brief Apply field-wise constraints
     */
    bool apply_constraints{true};

    /**
     * @brief Verbose output
     */
    bool verbose{false};
};

/**
 * @brief Statistics for block assembly
 */
struct BlockAssemblyStats {
    GlobalIndex num_cells{0};
    double total_seconds{0.0};

    // Per-block statistics
    std::map<BlockIndex, double> block_assembly_seconds;
    std::map<BlockIndex, GlobalIndex> block_nnz;
};

// ============================================================================
// Field Configuration
// ============================================================================

/**
 * @brief Configuration for a single field
 */
struct FieldConfig {
    FieldId id;                                 ///< Field identifier
    std::string name;                           ///< Human-readable name
    const spaces::FunctionSpace* space{nullptr}; ///< Function space
    const dofs::DofMap* dof_map{nullptr};        ///< DOF map for this field
    const constraints::AffineConstraints* constraints{nullptr}; ///< Field constraints

    // Field properties
    int components{1};                          ///< Number of components (e.g., 3 for velocity)
    bool is_pressure_like{false};               ///< For special handling (e.g., no BCs at outflow)
};

/**
 * @brief Block system configuration
 */
struct BlockSystemConfig {
    std::vector<FieldConfig> fields;

    /**
     * @brief Add a field
     */
    void addField(FieldId id, const std::string& name,
                  const spaces::FunctionSpace& space,
                  const dofs::DofMap& dof_map) {
        fields.push_back({id, name, &space, &dof_map, nullptr, 1, false});
    }

    /**
     * @brief Get field by ID
     */
    const FieldConfig* getField(FieldId id) const {
        for (const auto& f : fields) {
            if (f.id == id) return &f;
        }
        return nullptr;
    }

    /**
     * @brief Get number of fields
     */
    int numFields() const { return static_cast<int>(fields.size()); }
};

// ============================================================================
// Block Views
// ============================================================================

/**
 * @brief View into a block of the global system
 *
 * Provides matrix/vector operations scoped to a specific block.
 */
class BlockView {
public:
    /**
     * @brief Construct block view
     *
     * @param global_view Underlying global system view
     * @param row_offset Starting row in global system
     * @param col_offset Starting column in global system
     * @param num_rows Number of rows in this block
     * @param num_cols Number of columns in this block
     */
    BlockView(GlobalSystemView& global_view,
              GlobalIndex row_offset, GlobalIndex col_offset,
              GlobalIndex num_rows, GlobalIndex num_cols);

    /**
     * @brief Add matrix entries (local indices)
     */
    void addMatrixEntries(
        std::span<const GlobalIndex> local_rows,
        std::span<const GlobalIndex> local_cols,
        std::span<const Real> values);

    /**
     * @brief Add vector entries (local indices)
     */
    void addVectorEntries(
        std::span<const GlobalIndex> local_indices,
        std::span<const Real> values);

    /**
     * @brief Get block dimensions
     */
    [[nodiscard]] GlobalIndex numRows() const noexcept { return num_rows_; }
    [[nodiscard]] GlobalIndex numCols() const noexcept { return num_cols_; }

private:
    GlobalSystemView& global_view_;
    GlobalIndex row_offset_;
    GlobalIndex col_offset_;
    GlobalIndex num_rows_;
    GlobalIndex num_cols_;

    // Scratch for index translation
    mutable std::vector<GlobalIndex> translated_rows_;
    mutable std::vector<GlobalIndex> translated_cols_;
};

// ============================================================================
// Block Assembler
// ============================================================================

/**
 * @brief Assembler for block-structured multi-field systems
 *
 * BlockAssembler manages assembly of systems with multiple coupled fields.
 * It supports monolithic assembly (single matrix) or block-wise assembly
 * (separate matrices for each block).
 *
 * Usage (Monolithic):
 * @code
 *   BlockAssembler assembler;
 *   assembler.setMesh(mesh);
 *
 *   // Configure fields
 *   assembler.addField(0, "field_0", space_0, dofs_0);
 *   assembler.addField(1, "field_1", space_1, dofs_1);
 *
 *   // Assign per-block kernels
 *   assembler.setBlockKernel(0, 0, kernel_00);
 *   assembler.setBlockKernel(0, 1, kernel_01);
 *   assembler.setBlockKernel(1, 0, kernel_10);
 *   assembler.setBlockKernel(1, 1, kernel_11);
 *
 *   // Assemble monolithic system
 *   assembler.assembleSystem(matrix_view, rhs_view);
 * @endcode
 *
 * Usage (Block-wise):
 * @code
 *   BlockAssembler assembler;
 *   assembler.setOptions({.mode = BlockAssemblyMode::Block});
 *   // ... configure fields ...
 *
 *   // Assemble individual blocks
 *   assembler.assembleBlock(0, 0, A_00);
 *   assembler.assembleBlock(0, 1, A_01);
 *   assembler.assembleBlock(1, 0, A_10);
 *   assembler.assembleBlock(1, 1, A_11);
 *
 *   assembler.assembleFieldRhs(0, f_0);
 *   assembler.assembleFieldRhs(1, f_1);
 * @endcode
 */
class BlockAssembler {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    BlockAssembler();

    /**
     * @brief Construct with options
     */
    explicit BlockAssembler(const BlockAssemblerOptions& options);

    /**
     * @brief Destructor
     */
    ~BlockAssembler();

    /**
     * @brief Move constructor
     */
    BlockAssembler(BlockAssembler&& other) noexcept;

    /**
     * @brief Move assignment
     */
    BlockAssembler& operator=(BlockAssembler&& other) noexcept;

    // Non-copyable
    BlockAssembler(const BlockAssembler&) = delete;
    BlockAssembler& operator=(const BlockAssembler&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set mesh access
     */
    void setMesh(const IMeshAccess& mesh);

    /**
     * @brief Add a field to the block system
     *
     * @param id Field identifier
     * @param name Human-readable name
     * @param space Function space for this field
     * @param dof_map DOF map for this field
     */
    void addField(FieldId id, const std::string& name,
                  const spaces::FunctionSpace& space,
                  const dofs::DofMap& dof_map);

    /**
     * @brief Set field constraints
     */
    void setFieldConstraints(FieldId id, const constraints::AffineConstraints& constraints);

    /**
     * @brief Set block DOF map (for monolithic assembly)
     */
    void setBlockDofMap(const dofs::BlockDofMap& block_dof_map);

    /**
     * @brief Set options
     */
    void setOptions(const BlockAssemblerOptions& options);

    /**
     * @brief Get current options
     */
    [[nodiscard]] const BlockAssemblerOptions& getOptions() const noexcept {
        return options_;
    }

    /**
     * @brief Get block system configuration
     */
    [[nodiscard]] const BlockSystemConfig& getConfig() const noexcept {
        return config_;
    }

    // =========================================================================
    // Assembler Assignment (per-block)
    // =========================================================================

    /**
     * @brief Set assembler for a specific block (row_field, col_field)
     */
    void setBlockAssembler(FieldId row_field, FieldId col_field,
                           std::shared_ptr<Assembler> assembler);

    /**
     * @brief Set assembler for all blocks involving a field (row or column)
     */
    void setFieldAssembler(FieldId field, std::shared_ptr<Assembler> assembler);

    /**
     * @brief Set assembler for diagonal block only
     */
    void setDiagonalAssembler(FieldId field, std::shared_ptr<Assembler> assembler);

    /**
     * @brief Set default assembler for blocks without explicit assignment
     */
    void setDefaultAssembler(std::shared_ptr<Assembler> assembler);

    /**
     * @brief Get assembler for a block (assigned or default)
     */
    [[nodiscard]] Assembler& getBlockAssembler(FieldId row_field, FieldId col_field);

    /**
     * @brief Check if block has an explicitly assigned assembler
     */
    [[nodiscard]] bool hasBlockAssembler(FieldId row_field, FieldId col_field) const;

    // =========================================================================
    // Kernel Assignment (per-block / per-field)
    // =========================================================================

    /**
     * @brief Set kernel for a specific block
     */
    void setBlockKernel(FieldId row_field, FieldId col_field,
                        std::shared_ptr<AssemblyKernel> kernel);

    /**
     * @brief Set RHS kernel for a field
     */
    void setRhsKernel(FieldId field, std::shared_ptr<AssemblyKernel> kernel);

    /**
     * @brief Check if block has a kernel (non-zero block)
     */
    [[nodiscard]] bool hasBlockKernel(FieldId row_field, FieldId col_field) const;

    /**
     * @brief Get all non-zero block indices
     */
    [[nodiscard]] std::vector<BlockIndex> getNonZeroBlocks() const;

    // =========================================================================
    // State Propagation
    // =========================================================================

    /**
     * @brief Set current solution vector (monolithic/global indexing)
     */
    void setCurrentSolution(std::span<const Real> global_solution);

    /**
     * @brief Set current physical time
     */
    void setTime(Real time);

    /**
     * @brief Set current time step size dt
     */
    void setTimeStep(Real dt);

    // =========================================================================
    // Monolithic Assembly
    // =========================================================================

    /**
     * @brief Assemble complete monolithic system
     *
     * Assembles all blocks into a single matrix and all field RHS into
     * a single vector.
     *
     * @param matrix_view Global matrix
     * @param rhs_view Global RHS vector
     * @return Assembly statistics
     */
    BlockAssemblyStats assembleSystem(
        GlobalSystemView& matrix_view,
        GlobalSystemView& rhs_view);

    /**
     * @brief Assemble matrix only (all blocks)
     */
    BlockAssemblyStats assembleMatrix(GlobalSystemView& matrix_view);

    /**
     * @brief Assemble RHS only (all fields)
     */
    BlockAssemblyStats assembleRhs(GlobalSystemView& rhs_view);

    // =========================================================================
    // Block-wise Assembly
    // =========================================================================

    /**
     * @brief Assemble a single block
     *
     * @param row_field Test field
     * @param col_field Trial field
     * @param block_view Output matrix for this block
     * @return Assembly statistics
     */
    BlockAssemblyStats assembleBlock(
        FieldId row_field,
        FieldId col_field,
        GlobalSystemView& block_view);

    /**
     * @brief Assemble RHS for a single field
     *
     * @param field Field index
     * @param rhs_view Output RHS vector for this field
     * @return Assembly statistics
     */
    BlockAssemblyStats assembleFieldRhs(
        FieldId field,
        GlobalSystemView& rhs_view);

    // =========================================================================
    // Parallel Block Assembly
    // =========================================================================

    /**
     * @brief Assemble multiple blocks concurrently
     */
    BlockAssemblyStats assembleBlocksParallel(
        std::span<const BlockIndex> blocks,
        GlobalSystemView& matrix_view,
        GlobalSystemView* rhs_view = nullptr);

    // =========================================================================
    // Coupling Assembly
    // =========================================================================

    /**
     * @brief Assemble only coupling (off-diagonal) blocks
     *
     * Useful when diagonal blocks don't change but couplings need updating.
     *
     * @param matrix_view Global matrix (only off-diagonal blocks updated)
     * @return Assembly statistics
     */
    BlockAssemblyStats assembleCouplingBlocks(GlobalSystemView& matrix_view);

    /**
     * @brief Assemble only diagonal blocks
     *
     * @param matrix_view Global matrix (only diagonal blocks updated)
     * @return Assembly statistics
     */
    BlockAssemblyStats assembleDiagonalBlocks(GlobalSystemView& matrix_view);

    // =========================================================================
    // Selective Assembly
    // =========================================================================

    /**
     * @brief Assemble specific blocks by predicate
     */
    BlockAssemblyStats assembleBlocksIf(
        std::function<bool(FieldId row, FieldId col)> predicate,
        GlobalSystemView& matrix_view);

    // =========================================================================
    // Block Information
    // =========================================================================

    /**
     * @brief Get number of fields
     */
    [[nodiscard]] int numFields() const noexcept {
        return static_cast<int>(config_.fields.size());
    }

    /**
     * @brief Get block offsets in monolithic system
     *
     * @param row_field Row field
     * @param col_field Column field
     * @return (row_offset, col_offset) in global system
     */
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getBlockOffset(
        FieldId row_field, FieldId col_field) const;

    /**
     * @brief Get block size
     *
     * @param row_field Row field
     * @param col_field Column field
     * @return (num_rows, num_cols) for this block
     */
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getBlockSize(
        FieldId row_field, FieldId col_field) const;

    /**
     * @brief Get total system size
     */
    [[nodiscard]] GlobalIndex totalSize() const noexcept;

    // =========================================================================
    // Query
    // =========================================================================

    /**
     * @brief Check if assembler is configured
     */
    [[nodiscard]] bool isConfigured() const noexcept;

    /**
     * @brief Get last assembly statistics
     */
    [[nodiscard]] const BlockAssemblyStats& getLastStats() const noexcept {
        return last_stats_;
    }

private:
    // =========================================================================
    // Internal Implementation
    // =========================================================================

    /**
     * @brief Compute block offsets
     */
    void computeBlockOffsets();

    /**
     * @brief Assemble a single block (internal)
     */
    void assembleBlockInternal(
        FieldId row_field,
        FieldId col_field,
        GlobalSystemView& output_view,
        bool is_monolithic);

    /**
     * @brief Assemble field RHS (internal)
     */
    void assembleFieldRhsInternal(
        FieldId field,
        GlobalSystemView& rhs_view,
        bool is_monolithic);

    // =========================================================================
    // Data Members
    // =========================================================================

    BlockSystemConfig config_;
    const IMeshAccess* mesh_{nullptr};
    const dofs::BlockDofMap* block_dof_map_{nullptr};

    // Per-block assemblers: block (row_field, col_field) -> Assembler
    std::map<BlockIndex, std::shared_ptr<Assembler>> block_assemblers_;

    // Per-block kernels: block (row_field, col_field) -> AssemblyKernel
    std::map<BlockIndex, std::shared_ptr<AssemblyKernel>> block_kernels_;

    // Per-field RHS kernels: field -> AssemblyKernel
    std::map<FieldId, std::shared_ptr<AssemblyKernel>> rhs_kernels_;

    // Default assembler for blocks without explicit assignment
    std::shared_ptr<Assembler> default_assembler_;

    // Block offsets (for monolithic assembly)
    std::vector<GlobalIndex> field_offsets_;  // Size: num_fields + 1
    std::vector<GlobalIndex> field_sizes_;    // Size: num_fields

    // Cached state for propagation
    std::span<const Real> current_solution_{};
    Real time_{0.0};
    Real dt_{0.0};

    // Options
    BlockAssemblerOptions options_;

    // Statistics
    BlockAssemblyStats last_stats_;
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Create block assembler
 */
std::unique_ptr<BlockAssembler> createBlockAssembler(
    const BlockAssemblerOptions& options = {});

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_BLOCK_ASSEMBLER_H
