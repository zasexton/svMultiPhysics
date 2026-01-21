/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_DOFS_FIELDDOFMAP_H
#define SVMP_FE_DOFS_FIELDDOFMAP_H

/**
 * @file FieldDofMap.h
 * @brief Multi-field DOF management with component support
 *
 * The FieldDofMap handles multiple fields (e.g., velocity, pressure, temperature)
 * in a unified DOF numbering system. Key features:
 *  - Named fields with arbitrary numbers of components
 *  - Block structure information for block preconditioners
 *  - Component-level DOF access for selective BCs
 *  - SubspaceView generation for field/component extraction
 *
 * Example use cases:
 *  - Stokes/Navier-Stokes: velocity (3 components) + pressure (1 component)
 *  - FSI: displacement (3) + velocity (3) + pressure (1)
 *  - Multi-species transport: multiple scalar fields
 */

#include "DofMap.h"
#include "DofIndexSet.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <string>
#include <unordered_map>
#include <optional>
#include <memory>

namespace svmp {
namespace FE {

// Forward declarations
namespace spaces {
    class FunctionSpace;
}

namespace dofs {

// Forward declaration
class SubspaceView;

/**
 * @brief How a vector-valued field maps physical components to DOFs
 *
 * - ComponentWise: DOFs are stored per component (e.g., ProductSpace / H1 vector fields).
 * - VectorBasis: DOFs are coefficients on vector-valued basis functions (e.g., H(curl)/H(div)).
 */
enum class FieldComponentDofLayout : std::uint8_t {
    ComponentWise,
    VectorBasis
};

/**
 * @brief Field descriptor for multi-field systems
 */
struct FieldDescriptor {
    std::string name;                   ///< Field name (e.g., "velocity", "pressure")
    LocalIndex n_components{1};         ///< Number of components (e.g., 3 for velocity)
    GlobalIndex dof_offset{0};          ///< Starting DOF index for this field
    GlobalIndex n_dofs{0};              ///< Total DOFs for this field
    int block_index{0};                 ///< Block index in block systems
    FieldComponentDofLayout component_dof_layout{FieldComponentDofLayout::ComponentWise};

    // Function space information
    std::string space_type;             ///< Type of function space (e.g., "Lagrange")
    int polynomial_order{1};            ///< Polynomial order
};

/**
 * @brief Layout strategy for multi-field DOF numbering
 */
enum class FieldLayout : std::uint8_t {
    /**
     * @brief Interleaved layout
     *
     * DOFs for all fields at each node are consecutive:
     * [u0,v0,w0,p0], [u1,v1,w1,p1], ...
     *
     * Good for: point-block solvers, cache locality in assembly
     */
    Interleaved,

    /**
     * @brief Block layout
     *
     * DOFs for each field are grouped:
     * [u0,u1,...], [v0,v1,...], [w0,w1,...], [p0,p1,...]
     *
     * Good for: block preconditioners, field-wise solvers
     */
    Block,

    /**
     * @brief Field-block layout
     *
     * Fields are grouped, components interleaved within field:
     * [u0,v0,w0,u1,v1,w1,...], [p0,p1,...]
     *
     * Good for: mixed methods, vector-valued fields
     */
    FieldBlock
};

/**
 * @brief Multi-field DOF management class
 *
 * Manages DOF numbering for systems with multiple physics fields.
 * Provides mapping between field/component indices and global DOF indices.
 */
class FieldDofMap {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    FieldDofMap();
    ~FieldDofMap();

    // Move semantics
    FieldDofMap(FieldDofMap&&) noexcept;
    FieldDofMap& operator=(FieldDofMap&&) noexcept;

    // No copy (large data)
    FieldDofMap(const FieldDofMap&) = delete;
    FieldDofMap& operator=(const FieldDofMap&) = delete;

    // =========================================================================
    // Field Registration
    // =========================================================================

    /**
     * @brief Add a scalar field (1 component)
     *
     * @param name Field name
     * @param n_dofs Number of DOFs for this field
     * @return Field index for later reference
     */
    int addScalarField(const std::string& name, GlobalIndex n_dofs);

    /**
     * @brief Add a vector field (multiple components)
     *
     * @param name Field name
     * @param n_components Number of components (e.g., 3 for 3D velocity)
     * @param n_dofs_per_component DOFs per component
     * @return Field index
     */
    int addVectorField(const std::string& name, LocalIndex n_components,
                       GlobalIndex n_dofs_per_component);

    /**
     * @brief Add a vector-valued field with a vector-basis DOF layout (H(curl)/H(div))
     *
     * For these spaces, DOFs are not stored per component. Component extraction
     * APIs (getComponentDofs/componentToGlobal/getComponentOfDof) are not defined.
     */
    int addVectorBasisField(const std::string& name, LocalIndex value_dimension,
                            GlobalIndex n_dofs);

    /**
     * @brief Add a field with function space
     *
     * @param name Field name
     * @param space Function space defining the field
     * @param n_mesh_entities Number of mesh entities (vertices, cells, etc.)
     * @return Field index
     */
    int addField(const std::string& name, const spaces::FunctionSpace& space,
                 GlobalIndex n_mesh_entities);

    /**
     * @brief Set the DOF layout strategy
     *
     * @param layout Layout strategy
     * @throws FEException if already finalized
     */
    void setLayout(FieldLayout layout);

    /**
     * @brief Finalize the field DOF map
     *
     * Computes offsets and prepares for queries.
     */
    void finalize();

    // =========================================================================
    // Field Queries
    // =========================================================================

    /**
     * @brief Get number of fields
     */
    [[nodiscard]] std::size_t numFields() const noexcept { return fields_.size(); }

    /**
     * @brief Get field descriptor by index
     */
    [[nodiscard]] const FieldDescriptor& getField(std::size_t field_idx) const;

    /**
     * @brief Get field descriptor by name
     */
    [[nodiscard]] const FieldDescriptor& getField(const std::string& name) const;

    /**
     * @brief Get field index by name
     * @return Field index, or -1 if not found
     */
    [[nodiscard]] int getFieldIndex(const std::string& name) const noexcept;

    /**
     * @brief Check if field exists
     */
    [[nodiscard]] bool hasField(const std::string& name) const noexcept;

    /**
     * @brief Get all field names
     */
    [[nodiscard]] std::vector<std::string> getFieldNames() const;

    // =========================================================================
    // DOF Range Queries
    // =========================================================================

    /**
     * @brief Get DOF range for a field
     *
     * @param field_idx Field index
     * @return (start_dof, end_dof) exclusive range
     */
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getFieldDofRange(std::size_t field_idx) const;

    /**
     * @brief Get DOF range for a field by name
     */
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getFieldDofRange(const std::string& name) const;

    /**
     * @brief Get block offset for a field
     *
     * @param field_idx Field index
     * @return Block offset for block-structured matrices
     */
    [[nodiscard]] GlobalIndex getFieldOffset(std::size_t field_idx) const;

    /**
     * @brief Get total number of DOFs across all fields
     */
    [[nodiscard]] GlobalIndex totalDofs() const noexcept { return total_dofs_; }

    // =========================================================================
    // Component Queries
    // =========================================================================

    /**
     * @brief Get number of components for a field
     */
    [[nodiscard]] LocalIndex numComponents(std::size_t field_idx) const;

    /**
     * @brief Get number of components for a field by name
     */
    [[nodiscard]] LocalIndex numComponents(const std::string& name) const;

    /**
     * @brief Get DOFs for a specific component of a field
     *
     * @param field_idx Field index
     * @param component Component index (0 = first component)
     * @return IndexSet of DOFs for this component
     */
    [[nodiscard]] IndexSet getComponentDofs(std::size_t field_idx, LocalIndex component) const;

    /**
     * @brief Get DOFs for a specific component by field name
     */
    [[nodiscard]] IndexSet getComponentDofs(const std::string& name, LocalIndex component) const;

    /**
     * @brief Determine which field and component a DOF belongs to
     *
     * @param dof_id Global DOF index
     * @return (field_index, component_index), or nullopt if invalid
     */
    [[nodiscard]] std::optional<std::pair<int, LocalIndex>>
    getComponentOfDof(GlobalIndex dof_id) const;

    // =========================================================================
    // Subspace Views
    // =========================================================================

    /**
     * @brief Get a view into a single field
     *
     * @param field_idx Field index
     * @return SubspaceView for the field
     */
    [[nodiscard]] std::unique_ptr<SubspaceView> getFieldView(std::size_t field_idx) const;

    /**
     * @brief Get a view into a single field by name
     */
    [[nodiscard]] std::unique_ptr<SubspaceView> getFieldView(const std::string& name) const;

    /**
     * @brief Get a view into selected components of a field
     *
     * @param field_idx Field index
     * @param components Which components to include
     * @return SubspaceView for the selected components
     */
    [[nodiscard]] std::unique_ptr<SubspaceView> getComponentView(
        std::size_t field_idx, std::span<const LocalIndex> components) const;

    // =========================================================================
    // Layout Information
    // =========================================================================

    /**
     * @brief Get the DOF layout strategy
     */
    [[nodiscard]] FieldLayout layout() const noexcept { return layout_; }

    /**
     * @brief Check if layout is interleaved
     */
    [[nodiscard]] bool isInterleaved() const noexcept {
        return layout_ == FieldLayout::Interleaved;
    }

    /**
     * @brief Check if layout is block
     */
    [[nodiscard]] bool isBlock() const noexcept {
        return layout_ == FieldLayout::Block;
    }

    /**
     * @brief Get block sizes (for block layout)
     */
    [[nodiscard]] std::vector<GlobalIndex> getBlockSizes() const;

    /**
     * @brief Get block offsets (for block layout)
     */
    [[nodiscard]] std::vector<GlobalIndex> getBlockOffsets() const;

    // =========================================================================
    // DOF Mapping
    // =========================================================================

    /**
     * @brief Map field-local DOF to global DOF
     *
     * @param field_idx Field index
     * @param local_dof DOF index within the field
     * @return Global DOF index
     */
    [[nodiscard]] GlobalIndex fieldToGlobal(std::size_t field_idx, GlobalIndex local_dof) const;

    /**
     * @brief Map component-local DOF to global DOF
     *
     * @param field_idx Field index
     * @param component Component index
     * @param local_dof DOF index within the component
     * @return Global DOF index
     */
    [[nodiscard]] GlobalIndex componentToGlobal(std::size_t field_idx, LocalIndex component,
                                                 GlobalIndex local_dof) const;

    /**
     * @brief Map global DOF to field-local DOF
     *
     * @param dof_id Global DOF index
     * @return (field_idx, local_dof), or nullopt if invalid
     */
    [[nodiscard]] std::optional<std::pair<int, GlobalIndex>>
    globalToField(GlobalIndex dof_id) const;

    // =========================================================================
    // State
    // =========================================================================

    [[nodiscard]] bool isFinalized() const noexcept { return finalized_; }

private:
    // Check state
    void checkFinalized() const;
    void checkNotFinalized() const;

    // Compute offsets based on layout
    void computeOffsets();

    // Field storage
    std::vector<FieldDescriptor> fields_;
    std::unordered_map<std::string, std::size_t> name_to_index_;

    // Layout
    FieldLayout layout_{FieldLayout::Interleaved};

    // Computed offsets
    std::vector<GlobalIndex> field_offsets_;      // Start DOF for each field
    std::vector<GlobalIndex> component_offsets_;  // For block layout

    // Totals
    GlobalIndex total_dofs_{0};
    GlobalIndex total_components_{0};

    // State
    bool finalized_{false};
};

} // namespace dofs
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_DOFS_FIELDDOFMAP_H
