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
 * such as:
 * - Navier-Stokes: velocity-pressure coupling
 * - FSI: fluid-structure coupling
 * - Multiphysics: thermal-mechanical coupling
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
#include "AssemblyContext.h"
#include "AssemblyLoop.h"
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
// Block Kernel Interface
// ============================================================================

/**
 * @brief Kernel interface for block assembly
 *
 * Block kernels compute element contributions for multi-field systems.
 * Each field combination can have its own computation method.
 */
class IBlockKernel {
public:
    virtual ~IBlockKernel() = default;

    /**
     * @brief Compute block (i,j) element matrix
     *
     * @param context Assembly context
     * @param row_field Row (test) field index
     * @param col_field Column (trial) field index
     * @param output Output for local block matrix
     */
    virtual void computeBlock(
        AssemblyContext& context,
        FieldId row_field,
        FieldId col_field,
        KernelOutput& output) = 0;

    /**
     * @brief Compute field RHS contribution
     *
     * @param context Assembly context
     * @param field Field index
     * @param output Output for local RHS
     */
    virtual void computeRhs(
        AssemblyContext& context,
        FieldId field,
        KernelOutput& output) = 0;

    /**
     * @brief Check if block (i,j) is non-zero
     *
     * Allows skipping zero blocks in assembly.
     */
    [[nodiscard]] virtual bool hasBlock(FieldId row_field, FieldId col_field) const = 0;

    /**
     * @brief Get number of fields
     */
    [[nodiscard]] virtual int numFields() const = 0;

    /**
     * @brief Get required data for block computation
     */
    [[nodiscard]] virtual RequiredData getRequiredData(
        FieldId row_field, FieldId col_field) const
    {
        (void)row_field;
        (void)col_field;
        return RequiredData::BasisValues | RequiredData::BasisGradients;
    }
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
 *   assembler.addField(0, "velocity", velocity_space, velocity_dofs);
 *   assembler.addField(1, "pressure", pressure_space, pressure_dofs);
 *
 *   // Set kernel
 *   assembler.setKernel(stokes_kernel);
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
 *   assembler.assembleBlock(0, 0, A_uu);  // velocity-velocity
 *   assembler.assembleBlock(0, 1, A_up);  // velocity-pressure
 *   assembler.assembleBlock(1, 0, A_pu);  // pressure-velocity
 *   assembler.assembleBlock(1, 1, A_pp);  // pressure-pressure
 *
 *   assembler.assembleRhs(0, f_u);        // velocity RHS
 *   assembler.assembleRhs(1, f_p);        // pressure RHS
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
     * @brief Set the block kernel
     */
    void setKernel(IBlockKernel& kernel);

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
        GlobalSystemView& matrix_view,
        GlobalIndex row_offset,
        GlobalIndex col_offset);

    /**
     * @brief Assemble field RHS (internal)
     */
    void assembleFieldRhsInternal(
        FieldId field,
        GlobalSystemView& rhs_view,
        GlobalIndex offset);

    /**
     * @brief Get DOFs for cell and field
     */
    void getCellFieldDofs(
        GlobalIndex cell_id,
        FieldId field,
        std::vector<GlobalIndex>& dofs);

    // =========================================================================
    // Data Members
    // =========================================================================

    // Configuration
    BlockAssemblerOptions options_;
    BlockSystemConfig config_;
    const IMeshAccess* mesh_{nullptr};
    IBlockKernel* kernel_{nullptr};
    const dofs::BlockDofMap* block_dof_map_{nullptr};

    // Block offsets (for monolithic assembly)
    std::vector<GlobalIndex> field_offsets_;  // Size: num_fields + 1
    std::vector<GlobalIndex> field_sizes_;    // Size: num_fields

    // Assembly infrastructure
    std::unique_ptr<AssemblyLoop> loop_;

    // Thread-local storage
    std::vector<std::unique_ptr<AssemblyContext>> thread_contexts_;
    std::vector<KernelOutput> thread_outputs_;
    std::vector<std::vector<GlobalIndex>> thread_row_dofs_;
    std::vector<std::vector<GlobalIndex>> thread_col_dofs_;

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

/**
 * @brief Create simple 2-field block kernel
 *
 * Creates a block kernel for systems like Stokes (velocity-pressure).
 *
 * @param a_kernel Kernel for A block (velocity-velocity)
 * @param b_kernel Kernel for B block (velocity-pressure)
 * @param bt_kernel Kernel for B^T block (pressure-velocity)
 * @param c_kernel Kernel for C block (pressure-pressure, may be null)
 * @return Block kernel
 */
std::unique_ptr<IBlockKernel> createTwoFieldBlockKernel(
    AssemblyKernel& a_kernel,
    AssemblyKernel& b_kernel,
    AssemblyKernel* bt_kernel = nullptr,
    AssemblyKernel* c_kernel = nullptr);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_BLOCK_ASSEMBLER_H
