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

#ifndef SVMP_FE_SPARSITY_DG_SPARSITY_BUILDER_H
#define SVMP_FE_SPARSITY_DG_SPARSITY_BUILDER_H

/**
 * @file DGSparsityBuilder.h
 * @brief Specialized builder for DG/Interface/Face term sparsity patterns
 *
 * This header provides the DGSparsityBuilder class for constructing sparsity
 * patterns that arise from Discontinuous Galerkin (DG) methods, interior
 * penalty methods, and other face-based formulations. Unlike CG methods
 * where couplings arise from shared element nodes, DG methods create
 * additional couplings through face integrals.
 *
 * Key features:
 * - Cell coupling patterns (volume integrals, like CG)
 * - Face coupling patterns (interior face integrals between neighbors)
 * - Boundary face patterns (boundary condition terms)
 * - Configurable stencil types (compact, extended)
 * - Support for hybrid CG-DG patterns
 *
 * DG sparsity structure:
 * For interior face F between cells K+ and K-:
 * - DOFs from K+ couple with DOFs from K+ (self-coupling)
 * - DOFs from K+ couple with DOFs from K- (cross-coupling)
 * - DOFs from K- couple with DOFs from K+ (cross-coupling)
 * - DOFs from K- couple with DOFs from K- (self-coupling)
 *
 * Complexity notes:
 * - buildCellCouplings(): O(n_cells * dofs_per_cell^2)
 * - buildFaceCouplings(): O(n_faces * 2 * dofs_per_cell^2)
 * - Total: O(n_cells * dofs_per_cell^2 + n_faces * 4 * dofs_per_cell^2)
 *
 * @see SparsityBuilder for element-based CG patterns
 * @see SparsityPattern for the pattern representation
 */

#include "SparsityPattern.h"
#include "DistributedSparsityPattern.h"
#include "SparsityBuilder.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <functional>
#include <memory>
#include <optional>

namespace svmp {
namespace FE {
namespace sparsity {

/**
 * @brief Type of DG term contributing to sparsity
 */
enum class DGTermType : std::uint8_t {
    VolumeIntegral,    ///< Cell-local volume integrals (mass, stiffness)
    InteriorFace,      ///< Interior face integrals (jumps, averages)
    BoundaryFace,      ///< Boundary face integrals
    Penalty,           ///< Penalty terms (may have extended stencil)
    Stabilization,     ///< Stabilization terms
    All                ///< All term types
};

/**
 * @brief Stencil type for DG face couplings
 */
enum class DGStencilType : std::uint8_t {
    /**
     * @brief Compact stencil - couples only immediate neighbors
     *
     * For interior face between K+ and K-, couples:
     * - K+ DOFs with K+ and K- DOFs
     * - K- DOFs with K+ and K- DOFs
     */
    Compact,

    /**
     * @brief Extended stencil - includes face-of-face neighbors
     *
     * Used for some stabilization terms and high-order reconstruction.
     * Couples cells that share a vertex or edge through the face.
     */
    Extended,

    /**
     * @brief Node-based stencil - couples through shared nodes
     *
     * Similar to CG but includes DG-specific face terms.
     * Used for hybrid CG-DG formulations.
     */
    NodeBased
};

/**
 * @brief Configuration options for DG sparsity construction
 */
struct DGSparsityOptions {
    bool include_cell_couplings{true};     ///< Include volume integral couplings
    bool include_face_couplings{true};     ///< Include interior face couplings
    bool include_boundary_couplings{true}; ///< Include boundary face couplings
    DGStencilType stencil_type{DGStencilType::Compact}; ///< Stencil type
    bool symmetric_pattern{false};         ///< Build A + A^T pattern
    bool ensure_diagonal{true};            ///< Force diagonal entries
    bool deterministic{true};              ///< Ensure deterministic ordering
};

/**
 * @brief Abstract interface for face connectivity queries
 *
 * This interface provides the mesh connectivity information needed to
 * construct DG face couplings without a hard dependency on the Mesh module.
 */
class IFaceConnectivity {
public:
    virtual ~IFaceConnectivity() = default;

    /**
     * @brief Get number of interior faces
     */
    [[nodiscard]] virtual GlobalIndex getNumInteriorFaces() const = 0;

    /**
     * @brief Get number of boundary faces
     */
    [[nodiscard]] virtual GlobalIndex getNumBoundaryFaces() const = 0;

    /**
     * @brief Get cells adjacent to an interior face
     *
     * @param face_id Interior face index
     * @return Pair (cell_plus, cell_minus) of adjacent cell indices
     */
    [[nodiscard]] virtual std::pair<GlobalIndex, GlobalIndex>
        getInteriorFaceCells(GlobalIndex face_id) const = 0;

    /**
     * @brief Get cell adjacent to a boundary face
     *
     * @param face_id Boundary face index
     * @return Cell index adjacent to this boundary face
     */
    [[nodiscard]] virtual GlobalIndex getBoundaryFaceCell(GlobalIndex face_id) const = 0;

    /**
     * @brief Get boundary tag/marker for a boundary face
     *
     * @param face_id Boundary face index
     * @return Boundary tag (for selective boundary coupling)
     */
    [[nodiscard]] virtual int getBoundaryTag(GlobalIndex face_id) const = 0;
};

/**
 * @brief Abstract interface for DG DOF map queries
 *
 * Extends IDofMapQuery with DG-specific methods for face DOFs.
 */
class IDGDofMapQuery : public IDofMapQuery {
public:
    /**
     * @brief Get DOFs on a face (for face-based assembly)
     *
     * @param cell_id Cell index
     * @param local_face Local face index within cell
     * @return Span of global DOF indices on this face
     */
    [[nodiscard]] virtual std::span<const GlobalIndex>
        getFaceDofs(GlobalIndex cell_id, LocalIndex local_face) const = 0;

    /**
     * @brief Get number of faces per cell for given cell
     */
    [[nodiscard]] virtual LocalIndex getNumFacesPerCell(GlobalIndex cell_id) const = 0;

    /**
     * @brief Check if DOF map is DG (discontinuous)
     */
    [[nodiscard]] virtual bool isDG() const = 0;
};

/**
 * @brief Simple face connectivity implementation
 *
 * Stores face-cell connectivity as arrays. Useful for building
 * connectivity programmatically or from mesh data.
 */
class SimpleFaceConnectivity : public IFaceConnectivity {
public:
    SimpleFaceConnectivity() = default;

    /**
     * @brief Add an interior face
     *
     * @param cell_plus Plus-side cell index
     * @param cell_minus Minus-side cell index
     * @return Interior face index
     */
    GlobalIndex addInteriorFace(GlobalIndex cell_plus, GlobalIndex cell_minus);

    /**
     * @brief Add a boundary face
     *
     * @param cell Adjacent cell index
     * @param tag Boundary tag/marker
     * @return Boundary face index
     */
    GlobalIndex addBoundaryFace(GlobalIndex cell, int tag = 0);

    /**
     * @brief Clear all faces
     */
    void clear();

    // IFaceConnectivity interface
    [[nodiscard]] GlobalIndex getNumInteriorFaces() const override;
    [[nodiscard]] GlobalIndex getNumBoundaryFaces() const override;
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
        getInteriorFaceCells(GlobalIndex face_id) const override;
    [[nodiscard]] GlobalIndex getBoundaryFaceCell(GlobalIndex face_id) const override;
    [[nodiscard]] int getBoundaryTag(GlobalIndex face_id) const override;

private:
    std::vector<std::pair<GlobalIndex, GlobalIndex>> interior_faces_;
    std::vector<std::pair<GlobalIndex, int>> boundary_faces_; // (cell, tag)
};

/**
 * @brief Statistics about DG sparsity pattern
 */
struct DGSparsityStats {
    GlobalIndex n_cells{0};           ///< Number of cells
    GlobalIndex n_interior_faces{0};  ///< Number of interior faces
    GlobalIndex n_boundary_faces{0};  ///< Number of boundary faces
    GlobalIndex cell_couplings{0};    ///< NNZ from cell couplings
    GlobalIndex face_couplings{0};    ///< NNZ from face couplings
    GlobalIndex boundary_couplings{0}; ///< NNZ from boundary couplings
    GlobalIndex total_nnz{0};         ///< Total NNZ
};

/**
 * @brief Builder for DG sparsity patterns
 *
 * DGSparsityBuilder constructs sparsity patterns for discontinuous Galerkin
 * and other face-based finite element formulations. It handles:
 *
 * 1. Cell-local couplings (volume integrals)
 * 2. Interior face couplings (neighbor interactions)
 * 3. Boundary face couplings (boundary condition terms)
 *
 * Usage:
 * @code
 * DGSparsityBuilder builder;
 * builder.setDofMap(dg_dof_map);
 * builder.setFaceConnectivity(face_connectivity);
 *
 * // Build complete DG pattern
 * SparsityPattern pattern = builder.build();
 *
 * // Or build incrementally
 * SparsityPattern pattern(n_dofs, n_dofs);
 * builder.buildCellCouplings(pattern);
 * builder.buildFaceCouplings(pattern);
 * builder.buildBoundaryFaceCouplings(pattern);
 * pattern.finalize();
 * @endcode
 *
 * @note For mixed CG-DG problems, use SparsityBuilder for CG parts and
 *       DGSparsityBuilder for DG parts, then combine with patternUnion().
 */
class DGSparsityBuilder {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    DGSparsityBuilder() = default;

    /**
     * @brief Construct with DG DOF map and face connectivity
     *
     * @param dof_map DG DOF map
     * @param face_connectivity Face-cell connectivity
     */
    DGSparsityBuilder(std::shared_ptr<IDGDofMapQuery> dof_map,
                      std::shared_ptr<IFaceConnectivity> face_connectivity);

    /// Destructor
    ~DGSparsityBuilder() = default;

    // Non-copyable (shared_ptr members)
    DGSparsityBuilder(const DGSparsityBuilder&) = delete;
    DGSparsityBuilder& operator=(const DGSparsityBuilder&) = delete;

    // Movable
    DGSparsityBuilder(DGSparsityBuilder&&) = default;
    DGSparsityBuilder& operator=(DGSparsityBuilder&&) = default;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set the DG DOF map
     */
    void setDofMap(std::shared_ptr<IDGDofMapQuery> dof_map);

    /**
     * @brief Set standard DOF map (wraps as IDGDofMapQuery)
     */
    void setDofMap(std::shared_ptr<IDofMapQuery> dof_map);

    /**
     * @brief Set face connectivity
     */
    void setFaceConnectivity(std::shared_ptr<IFaceConnectivity> connectivity);

    /**
     * @brief Set build options
     */
    void setOptions(const DGSparsityOptions& options) {
        options_ = options;
    }

    /**
     * @brief Get current options
     */
    [[nodiscard]] const DGSparsityOptions& getOptions() const noexcept {
        return options_;
    }

    // =========================================================================
    // Building methods
    // =========================================================================

    /**
     * @brief Build complete DG sparsity pattern
     *
     * Includes cell, face, and boundary couplings based on options.
     *
     * @return Finalized sparsity pattern
     * @throws FEException if not properly configured
     */
    [[nodiscard]] SparsityPattern build();

    /**
     * @brief Build pattern for specific term types
     *
     * @param term_types Bitmask of DGTermType to include
     * @return Finalized sparsity pattern
     */
    [[nodiscard]] SparsityPattern build(DGTermType term_type);

    /**
     * @brief Add cell-local couplings to pattern
     *
     * Adds DOF couplings within each cell (volume integrals).
     *
     * @param pattern Pattern to augment (must be in Building state)
     */
    void buildCellCouplings(SparsityPattern& pattern);

    /**
     * @brief Add interior face couplings to pattern
     *
     * Adds DOF couplings across interior faces (face integrals).
     *
     * @param pattern Pattern to augment
     */
    void buildFaceCouplings(SparsityPattern& pattern);

    /**
     * @brief Add interior face couplings for specific faces
     *
     * @param pattern Pattern to augment
     * @param face_ids Interior face indices to process
     */
    void buildFaceCouplings(SparsityPattern& pattern,
                            std::span<const GlobalIndex> face_ids);

    /**
     * @brief Add boundary face couplings to pattern
     *
     * Adds DOF couplings for boundary face terms.
     *
     * @param pattern Pattern to augment
     */
    void buildBoundaryFaceCouplings(SparsityPattern& pattern);

    /**
     * @brief Add boundary face couplings for specific boundary tag
     *
     * @param pattern Pattern to augment
     * @param boundary_tag Boundary marker to process
     */
    void buildBoundaryFaceCouplings(SparsityPattern& pattern, int boundary_tag);

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Get statistics from last build
     */
    [[nodiscard]] const DGSparsityStats& getLastStats() const noexcept {
        return last_stats_;
    }

    /**
     * @brief Estimate NNZ for DG pattern
     *
     * @param n_cells Number of cells
     * @param dofs_per_cell DOFs per cell
     * @param avg_neighbors Average neighbors per cell
     * @return Estimated total NNZ
     */
    [[nodiscard]] static GlobalIndex estimateNnz(
        GlobalIndex n_cells,
        GlobalIndex dofs_per_cell,
        double avg_neighbors = 4.0);

private:
    // Validation
    void validateConfiguration() const;

    // Internal helpers
    void addCellCoupling(SparsityPattern& pattern, GlobalIndex cell_id);
    void addFaceCoupling(SparsityPattern& pattern,
                         GlobalIndex cell_plus, GlobalIndex cell_minus);
    void addBoundaryCoupling(SparsityPattern& pattern, GlobalIndex cell_id);

    // Configuration
    std::shared_ptr<IDGDofMapQuery> dof_map_;
    std::shared_ptr<IFaceConnectivity> face_connectivity_;
    DGSparsityOptions options_;

    // Statistics
    mutable DGSparsityStats last_stats_;
};

/**
 * @brief Adapter to wrap IDofMapQuery as IDGDofMapQuery
 *
 * For use when a standard DOF map is used with DGSparsityBuilder.
 * Face DOFs are derived from cell DOFs (all cell DOFs are face DOFs).
 */
class DGDofMapAdapter : public IDGDofMapQuery {
public:
    explicit DGDofMapAdapter(std::shared_ptr<IDofMapQuery> dof_map);

    // IDofMapQuery interface
    [[nodiscard]] GlobalIndex getNumDofs() const override;
    [[nodiscard]] GlobalIndex getNumLocalDofs() const override;
    [[nodiscard]] std::span<const GlobalIndex> getCellDofs(GlobalIndex cell_id) const override;
    [[nodiscard]] GlobalIndex getNumCells() const override;
    [[nodiscard]] bool isOwnedDof(GlobalIndex dof) const override;
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getOwnedRange() const override;

    // IDGDofMapQuery interface
    [[nodiscard]] std::span<const GlobalIndex>
        getFaceDofs(GlobalIndex cell_id, LocalIndex local_face) const override;
    [[nodiscard]] LocalIndex getNumFacesPerCell(GlobalIndex cell_id) const override;
    [[nodiscard]] bool isDG() const override;

private:
    std::shared_ptr<IDofMapQuery> dof_map_;
    mutable std::vector<GlobalIndex> face_dofs_cache_;
};

// ============================================================================
// Convenience functions
// ============================================================================

/**
 * @brief Build DG sparsity pattern (convenience function)
 *
 * @param dof_map DG DOF map
 * @param face_connectivity Face-cell connectivity
 * @param options Build options
 * @return Finalized sparsity pattern
 */
[[nodiscard]] inline SparsityPattern buildDGPattern(
    std::shared_ptr<IDGDofMapQuery> dof_map,
    std::shared_ptr<IFaceConnectivity> face_connectivity,
    const DGSparsityOptions& options = DGSparsityOptions{})
{
    DGSparsityBuilder builder(std::move(dof_map), std::move(face_connectivity));
    builder.setOptions(options);
    return builder.build();
}

/**
 * @brief Combine CG and DG patterns for hybrid methods
 *
 * @param cg_pattern Pattern from CG parts
 * @param dg_pattern Pattern from DG parts
 * @return Combined pattern
 */
[[nodiscard]] inline SparsityPattern combineHybridPattern(
    const SparsityPattern& cg_pattern,
    const SparsityPattern& dg_pattern)
{
    return patternUnion(cg_pattern, dg_pattern);
}

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_DG_SPARSITY_BUILDER_H
