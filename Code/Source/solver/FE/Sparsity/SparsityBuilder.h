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

#ifndef SVMP_FE_SPARSITY_SPARSITY_BUILDER_H
#define SVMP_FE_SPARSITY_SPARSITY_BUILDER_H

/**
 * @file SparsityBuilder.h
 * @brief Build sparsity patterns from mesh connectivity and DOF maps
 *
 * This header provides the SparsityBuilder class for constructing sparsity
 * patterns from mesh topology and DOF (degree of freedom) mappings. It is the
 * primary interface for creating FEM matrix sparsity patterns.
 *
 * Key features:
 * - Element-wise DOF connectivity traversal (CG patterns via DofMap)
 * - Support for rectangular operators (different row/col DOF maps)
 * - Multi-field coupling support (block-diagonal / custom block pairs)
 * - Ghost DOF handling for parallel assembly
 * - Deterministic pattern construction
 *
 * The builder operates by iterating over mesh elements and inserting
 * couplings between DOFs that interact through element integrals.
 *
 * Complexity notes:
 * - build(): O(n_elements * dofs_per_elem^2 * log(row_nnz))
 * - Memory: O(n_rows * avg_couplings_per_row) during building
 *
 * @see SparsityPattern for the output pattern representation
 * @see DistributedSparsityPattern for MPI-parallel patterns
 * @see DGSparsityBuilder for face-based DG couplings (Phase 2)
 */

#include "SparsityPattern.h"
#include "DistributedSparsityPattern.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"
#include "Dofs/DofMap.h"

#include <vector>
#include <span>
#include <functional>
#include <memory>
#include <optional>

// Forward declarations from other FE modules (avoid hard dependency)
namespace svmp {
namespace FE {
namespace dofs {
class BlockDofMap;
class FieldDofMap;
} // namespace dofs
} // namespace FE
}

namespace svmp {
namespace FE {
namespace sparsity {

/**
 * @brief Options for sparsity pattern construction
 */
struct SparsityBuildOptions {
    bool ensure_diagonal{true};       ///< Force diagonal entries for all rows
    bool ensure_non_empty_rows{true}; ///< Force at least one entry per row
    bool symmetric_pattern{false};    ///< Build A + A^T pattern (for symmetric matrices)
    bool include_ghost_rows{true};    ///< Include couplings to non-owned (ghost) DOFs in distributed patterns
    bool deterministic{true};         ///< Ensure deterministic ordering (sorted cols)
};

/**
 * @brief Coupling mode for multi-field sparsity patterns
 */
enum class CouplingMode : std::uint8_t {
    Full,       ///< All fields couple with all fields
    Diagonal,   ///< Each field couples only with itself
    Custom      ///< User-specified coupling pairs
};

/**
 * @brief Field coupling specification for multi-field problems
 *
 * Represents which DOF fields should couple in the sparsity pattern.
 * For example, in Stokes flow, velocity DOFs couple with both velocity
 * and pressure DOFs, while pressure may only couple with velocity (for
 * the divergence constraint).
 */
struct FieldCoupling {
    FieldId row_field{0};    ///< Row field identifier
    FieldId col_field{0};    ///< Column field identifier
    bool bidirectional{false}; ///< If true, also adds (col_field, row_field) coupling
};

/**
 * @brief Abstract interface for DOF map queries
 *
 * This interface allows the SparsityBuilder to work with different DOF
 * mapping implementations without a hard dependency on specific classes.
 * It provides the minimal query interface needed for sparsity construction.
 */
class IDofMapQuery {
public:
    virtual ~IDofMapQuery() = default;

    /**
     * @brief Get total number of DOFs (global)
     */
    [[nodiscard]] virtual GlobalIndex getNumDofs() const = 0;

    /**
     * @brief Get number of locally owned DOFs (for parallel)
     */
    [[nodiscard]] virtual GlobalIndex getNumLocalDofs() const = 0;

    /**
     * @brief Get DOFs for a cell/element
     *
     * @param cell_id Local cell index
     * @return Span of global DOF indices for this cell
     */
    [[nodiscard]] virtual std::span<const GlobalIndex> getCellDofs(GlobalIndex cell_id) const = 0;

    /**
     * @brief Get number of cells/elements
     */
    [[nodiscard]] virtual GlobalIndex getNumCells() const = 0;

    /**
     * @brief Check if a DOF is locally owned (for parallel)
     */
    [[nodiscard]] virtual bool isOwnedDof(GlobalIndex dof) const = 0;

    /**
     * @brief Get ownership range [first, last) of locally owned DOFs
     *
     * Returns (0, getNumDofs()) for serial, actual range for parallel.
     */
    [[nodiscard]] virtual std::pair<GlobalIndex, GlobalIndex> getOwnedRange() const = 0;
};

/**
 * @brief Abstract interface for mapping DOFs to field IDs
 *
 * This enables multi-field coupling rules (block-diagonal, custom block coupling)
 * without requiring a specific multi-field DOF layout implementation.
 *
 * Typical adapters include:
 * - dofs::BlockDofMap (block ranges + coupling matrix)
 * - dofs::FieldDofMap (field ranges)
 */
class IDofFieldQuery {
public:
    virtual ~IDofFieldQuery() = default;

    /**
     * @brief Get the field ID associated with a global DOF
     *
     * @return FieldId for the DOF, or INVALID_FIELD_ID if unknown.
     */
    [[nodiscard]] virtual FieldId getFieldId(GlobalIndex dof) const = 0;

    /**
     * @brief Get number of fields (if known)
     */
    [[nodiscard]] virtual std::size_t numFields() const = 0;
};

/**
 * @brief Adapter to wrap dofs::DofMap as IDofMapQuery
 */
class DofMapAdapter : public IDofMapQuery {
public:
    explicit DofMapAdapter(const dofs::DofMap& dof_map)
        : dof_map_(dof_map) {}

    [[nodiscard]] GlobalIndex getNumDofs() const override {
        return dof_map_.getNumDofs();
    }

    [[nodiscard]] GlobalIndex getNumLocalDofs() const override {
        return dof_map_.getNumLocalDofs();
    }

    [[nodiscard]] std::span<const GlobalIndex> getCellDofs(GlobalIndex cell_id) const override {
        return dof_map_.getCellDofs(cell_id);
    }

    [[nodiscard]] GlobalIndex getNumCells() const override {
        return dof_map_.getNumCells();
    }

    [[nodiscard]] bool isOwnedDof(GlobalIndex dof) const override {
        return dof_map_.isOwnedDof(dof);
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getOwnedRange() const override {
        GlobalIndex n_dofs = dof_map_.getNumDofs();
        if (n_dofs <= 0) {
            return {0, 0};
        }

        GlobalIndex n_local = dof_map_.getNumLocalDofs();
        if (n_local == n_dofs) {
            return {0, n_dofs};
        }

        GlobalIndex first_owned = -1;
        GlobalIndex last_owned = -1;
        bool exited_owned_region = false;

        for (GlobalIndex dof = 0; dof < n_dofs; ++dof) {
            const bool owned = dof_map_.isOwnedDof(dof);
            if (owned) {
                FE_CHECK_ARG(!exited_owned_region,
                             "DofMapAdapter::getOwnedRange requires contiguous ownership");
                if (first_owned < 0) {
                    first_owned = dof;
                }
                last_owned = dof + 1;
            } else if (first_owned >= 0) {
                exited_owned_region = true;
            }
        }

        if (first_owned < 0) {
            FE_CHECK_ARG(n_local == 0, "DofMapAdapter ownership mismatch: no owned DOFs detected");
            return {0, 0};
        }

        const GlobalIndex computed_local = last_owned - first_owned;
        if (n_local > 0) {
            FE_CHECK_ARG(computed_local == n_local,
                         "DofMapAdapter ownership mismatch: getNumLocalDofs()=" +
                             std::to_string(n_local) + " but inferred range size=" +
                             std::to_string(computed_local));
        }
        return {first_owned, last_owned};
    }

private:
    const dofs::DofMap& dof_map_;
};

/**
 * @brief Builder for creating sparsity patterns from mesh and DOF data
 *
 * SparsityBuilder traverses mesh elements and constructs the sparsity pattern
 * by adding couplings between DOFs that share element support. It supports:
 *
 * - Square patterns: same DOF map for rows and columns
 * - Rectangular patterns: different row/col DOF maps
 * - Multi-field problems: selective field couplings
 * - Parallel assembly: ghost DOF handling
 *
 * Usage for simple single-field case:
 * @code
 * SparsityBuilder builder;
 * builder.setRowDofMap(dof_map);
 * builder.setColDofMap(dof_map);  // Same for square
 * SparsityPattern pattern = builder.build();
 * @endcode
 *
 * Usage for multi-field rectangular case:
 * @code
 * SparsityBuilder builder;
 * builder.setRowDofMap(velocity_dofs);
 * builder.setColDofMap(pressure_dofs);
 * builder.setRowFieldMap(block_map);  // maps DOF -> field ID
 * builder.setColFieldMap(block_map);  // (or omit for square patterns)
 * builder.addCoupling(0, 0);          // velocity-velocity
 * builder.addCoupling(0, 1);          // velocity-pressure
 * SparsityPattern pattern = builder.build();
 * @endcode
 *
 * @note This builder constructs CG (continuous Galerkin) patterns based on
 *       element connectivity. For DG face couplings, use DGSparsityBuilder.
 */
class SparsityBuilder {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    SparsityBuilder() = default;

    /**
     * @brief Construct with DOF map for square pattern
     *
     * @param dof_map DOF map used for both rows and columns
     */
    explicit SparsityBuilder(const dofs::DofMap& dof_map);

    /**
     * @brief Construct with DOF map interface for square pattern
     *
     * @param dof_map_query DOF map query interface
     */
    explicit SparsityBuilder(std::shared_ptr<IDofMapQuery> dof_map_query);

    /// Destructor
    ~SparsityBuilder() = default;

    // Non-copyable due to shared_ptr members
    SparsityBuilder(const SparsityBuilder&) = delete;
    SparsityBuilder& operator=(const SparsityBuilder&) = delete;

    // Movable
    SparsityBuilder(SparsityBuilder&&) = default;
    SparsityBuilder& operator=(SparsityBuilder&&) = default;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set the row DOF map
     *
     * @param dof_map DOF map for row indices
     */
    void setRowDofMap(const dofs::DofMap& dof_map);

    /**
     * @brief Set the row DOF map via interface
     *
     * @param dof_map_query DOF map query interface
     */
    void setRowDofMap(std::shared_ptr<IDofMapQuery> dof_map_query);

    /**
     * @brief Set the column DOF map
     *
     * @param dof_map DOF map for column indices
     *
     * If not set, row DOF map is used (square pattern).
     */
    void setColDofMap(const dofs::DofMap& dof_map);

    /**
     * @brief Set the column DOF map via interface
     *
     * @param dof_map_query DOF map query interface
     */
    void setColDofMap(std::shared_ptr<IDofMapQuery> dof_map_query);

    /**
     * @brief Set build options
     *
     * @param options Build configuration options
     */
    void setOptions(const SparsityBuildOptions& options) {
        options_ = options;
    }

    /**
     * @brief Get current build options
     */
    [[nodiscard]] const SparsityBuildOptions& getOptions() const noexcept {
        return options_;
    }

    /**
     * @brief Set coupling mode for multi-field problems
     *
     * @param mode Coupling mode (Full, Diagonal, or Custom)
     */
    void setCouplingMode(CouplingMode mode) {
        coupling_mode_ = mode;
    }

    /**
     * @brief Add a field coupling (Custom mode)
     *
     * @param row_field Row field identifier
     * @param col_field Column field identifier
     * @param bidirectional If true, also adds reverse coupling
     */
    void addCoupling(FieldId row_field, FieldId col_field, bool bidirectional = false);

    /**
     * @brief Add a field coupling specification
     *
     * @param coupling Coupling specification
     */
    void addCoupling(const FieldCoupling& coupling);

    // -------------------------------------------------------------------------
    // Field mapping for multi-field coupling (optional)
    // -------------------------------------------------------------------------

    /**
     * @brief Set a field map for row DOFs (multi-field coupling support)
     */
    void setRowFieldMap(std::shared_ptr<IDofFieldQuery> field_query);

    /**
     * @brief Set a field map for column DOFs (multi-field coupling support)
     *
     * If not set, the row field map is used.
     */
    void setColFieldMap(std::shared_ptr<IDofFieldQuery> field_query);

    /**
     * @brief Convenience: set row field map from dofs::BlockDofMap
     */
    void setRowFieldMap(const dofs::BlockDofMap& block_map);

    /**
     * @brief Convenience: set column field map from dofs::BlockDofMap
     */
    void setColFieldMap(const dofs::BlockDofMap& block_map);

    /**
     * @brief Convenience: set row field map from dofs::FieldDofMap
     */
    void setRowFieldMap(const dofs::FieldDofMap& field_map);

    /**
     * @brief Convenience: set column field map from dofs::FieldDofMap
     */
    void setColFieldMap(const dofs::FieldDofMap& field_map);

    /**
     * @brief Populate Custom couplings from a dofs::BlockDofMap coupling matrix
     *
     * Interprets any non-None coupling as allowing (i,j). If coupling is TwoWay
     * or Full, also enables (j,i).
     *
     * Also sets CouplingMode::Custom.
     */
    void setCouplingsFromBlockDofMap(const dofs::BlockDofMap& block_map);

    /**
     * @brief Clear all custom field couplings
     */
    void clearCouplings() {
        field_couplings_.clear();
    }

    // =========================================================================
    // Building
    // =========================================================================

    /**
     * @brief Build the sparsity pattern
     *
     * Constructs a SparsityPattern by iterating over all cells and adding
     * couplings between DOFs that share element support.
     *
     * @return Finalized sparsity pattern
     * @throws FEException if DOF maps not configured or invalid
     *
     * Complexity: O(n_cells * dofs_per_cell^2 * log(row_nnz))
     */
    [[nodiscard]] SparsityPattern build();

    /**
     * @brief Build pattern for a subset of cells
     *
     * @param cell_ids Cell indices to include
     * @return Finalized sparsity pattern
     *
     * Useful for building patterns for boundary conditions or
     * domain decomposition.
     */
    [[nodiscard]] SparsityPattern build(std::span<const GlobalIndex> cell_ids);

    /**
     * @brief Build pattern with custom DOF getter
     *
     * @param n_rows Total number of rows
     * @param n_cols Total number of columns
     * @param n_elements Number of elements to process
     * @param get_row_dofs Function returning row DOFs for element i
     * @param get_col_dofs Function returning col DOFs for element i
     * @return Finalized sparsity pattern
     *
     * Low-level interface for maximum flexibility.
     */
    [[nodiscard]] SparsityPattern build(
        GlobalIndex n_rows,
        GlobalIndex n_cols,
        GlobalIndex n_elements,
        std::function<std::span<const GlobalIndex>(GlobalIndex)> get_row_dofs,
        std::function<std::span<const GlobalIndex>(GlobalIndex)> get_col_dofs);

    // =========================================================================
    // Convenience static methods
    // =========================================================================

    /**
     * @brief Build pattern from a single DOF map (square pattern)
     *
     * @param dof_map DOF map for both rows and columns
     * @param options Build options
     * @return Finalized sparsity pattern
     */
    [[nodiscard]] static SparsityPattern buildFromDofMap(
        const dofs::DofMap& dof_map,
        const SparsityBuildOptions& options = SparsityBuildOptions{});

    /**
     * @brief Build rectangular pattern from two DOF maps
     *
     * @param row_dof_map DOF map for rows
     * @param col_dof_map DOF map for columns
     * @param options Build options
     * @return Finalized sparsity pattern
     */
    [[nodiscard]] static SparsityPattern buildFromDofMaps(
        const dofs::DofMap& row_dof_map,
        const dofs::DofMap& col_dof_map,
        const SparsityBuildOptions& options = SparsityBuildOptions{});

    /**
     * @brief Build pattern from element-DOF connectivity arrays
     *
     * @param n_rows Total number of rows
     * @param n_cols Total number of columns
     * @param n_elements Number of elements
     * @param elem_row_offsets CSR-style offsets for row DOFs per element
     * @param elem_row_dofs Flat array of row DOFs
     * @param elem_col_offsets CSR-style offsets for col DOFs per element
     * @param elem_col_dofs Flat array of col DOFs
     * @param options Build options
     * @return Finalized sparsity pattern
     *
     * Lowest-level interface, no DOF map dependency.
     */
    [[nodiscard]] static SparsityPattern buildFromArrays(
        GlobalIndex n_rows,
        GlobalIndex n_cols,
        GlobalIndex n_elements,
        std::span<const GlobalIndex> elem_row_offsets,
        std::span<const GlobalIndex> elem_row_dofs,
        std::span<const GlobalIndex> elem_col_offsets,
        std::span<const GlobalIndex> elem_col_dofs,
        const SparsityBuildOptions& options = SparsityBuildOptions{});

    /**
     * @brief Build pattern from element-DOF connectivity (square, same DOFs)
     *
     * @param n_dofs Total number of DOFs
     * @param n_elements Number of elements
     * @param elem_offsets CSR-style offsets for DOFs per element
     * @param elem_dofs Flat array of DOFs
     * @param options Build options
     * @return Finalized sparsity pattern
     */
    [[nodiscard]] static SparsityPattern buildFromArrays(
        GlobalIndex n_dofs,
        GlobalIndex n_elements,
        std::span<const GlobalIndex> elem_offsets,
        std::span<const GlobalIndex> elem_dofs,
        const SparsityBuildOptions& options = SparsityBuildOptions{});

private:
    // Validate configuration before building
    void validateConfiguration() const;

    // Get effective column DOF map (row map if col map not set)
    [[nodiscard]] IDofMapQuery* getEffectiveColDofMap() const;

    // Get effective column field map (row map if col map not set)
    [[nodiscard]] IDofFieldQuery* getEffectiveColFieldMap() const;

    // DOF map adapters
    std::shared_ptr<IDofMapQuery> row_dof_map_;
    std::shared_ptr<IDofMapQuery> col_dof_map_;

    // Optional field mapping for coupling rules
    std::shared_ptr<IDofFieldQuery> row_field_map_;
    std::shared_ptr<IDofFieldQuery> col_field_map_;

    // Build options
    SparsityBuildOptions options_;

    // Multi-field coupling
    CouplingMode coupling_mode_{CouplingMode::Full};
    std::vector<FieldCoupling> field_couplings_;
};

/**
 * @brief Builder for distributed sparsity patterns
 *
 * Extends SparsityBuilder to produce DistributedSparsityPattern with
 * proper diag/offdiag separation for MPI-parallel matrices.
 */
class DistributedSparsityBuilder {
public:
    /**
     * @brief Default constructor
     */
    DistributedSparsityBuilder() = default;

    /**
     * @brief Construct with DOF map and ownership information
     *
     * @param dof_map DOF map for both rows and columns
     * @param first_owned_dof First globally owned DOF index
     * @param n_owned_dofs Number of locally owned DOFs
     * @param global_n_dofs Total global DOF count
     */
    DistributedSparsityBuilder(const dofs::DofMap& dof_map,
                               GlobalIndex first_owned_dof,
                               GlobalIndex n_owned_dofs,
                               GlobalIndex global_n_dofs);

    /**
     * @brief Set row ownership range
     *
     * @param first_owned First owned row index
     * @param n_owned Number of owned rows
     * @param global_n_rows Total global rows
     */
    void setRowOwnership(GlobalIndex first_owned, GlobalIndex n_owned, GlobalIndex global_n_rows);

    /**
     * @brief Set column ownership range
     *
     * @param first_owned First owned column index
     * @param n_owned Number of owned columns
     * @param global_n_cols Total global columns
     */
    void setColOwnership(GlobalIndex first_owned, GlobalIndex n_owned, GlobalIndex global_n_cols);

    /**
     * @brief Set row DOF map
     */
    void setRowDofMap(const dofs::DofMap& dof_map);

    /**
     * @brief Set row DOF map via interface
     */
    void setRowDofMap(std::shared_ptr<IDofMapQuery> dof_map_query);

    /**
     * @brief Set column DOF map
     */
    void setColDofMap(const dofs::DofMap& dof_map);

    /**
     * @brief Set column DOF map via interface
     */
    void setColDofMap(std::shared_ptr<IDofMapQuery> dof_map_query);

    /**
     * @brief Set build options
     */
    void setOptions(const SparsityBuildOptions& options) {
        options_ = options;
    }

    /**
     * @brief Build distributed sparsity pattern
     *
     * @return Finalized distributed sparsity pattern with diag/offdiag split
     * @throws FEException if ownership not configured
     */
    [[nodiscard]] DistributedSparsityPattern build();

    /**
     * @brief Build for subset of cells
     *
     * @param cell_ids Local cell indices to process
     * @return Finalized distributed sparsity pattern
     */
    [[nodiscard]] DistributedSparsityPattern build(std::span<const GlobalIndex> cell_ids);

private:
    // DOF maps
    std::shared_ptr<IDofMapQuery> row_dof_map_;
    std::shared_ptr<IDofMapQuery> col_dof_map_;

    // Ownership information
    IndexRange owned_rows_;
    IndexRange owned_cols_;
    GlobalIndex global_rows_{0};
    GlobalIndex global_cols_{0};

    // Build options
    SparsityBuildOptions options_;
};

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_SPARSITY_BUILDER_H
