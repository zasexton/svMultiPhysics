#ifndef SVMP_FE_SYSTEMS_FE_QUANTITY_DEFINITION_H
#define SVMP_FE_SYSTEMS_FE_QUANTITY_DEFINITION_H

/**
 * @file FEQuantityDefinition.h
 * @brief Definition types for FE-backed auxiliary input quantities.
 *
 * An `FEQuantityDefinition` describes one FE-backed auxiliary input:
 * - what kind of quantity it is (sampled field, boundary integral, etc.)
 * - what FE fields it references
 * - what shape (scalar/vector/tensor) the result has
 * - what region/marker/reduction applies
 * - whether explicit evaluation and monolithic linearization are supported
 *
 * Definitions are stored in `FEQuantityRegistry` and referenced from
 * `AuxiliaryInputHandle`.  The definition layer is separate from the
 * `AuxiliaryInputRegistry` (which manages evaluated numeric values and
 * dependency ordering).
 *
 * All types in this file are physics-agnostic.
 */

#include "Core/Types.h"

#include "Forms/FormExpr.h"

#include <cstdint>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ============================================================================
//  FE quantity kind
// ============================================================================

/**
 * @brief What kind of FE-backed quantity this is.
 */
enum class FEQuantityKind : std::uint8_t {
    /// Direct DOF lookup at mesh vertices (Lagrange elements).
    SampledField,

    /// Quadrature-weighted integral over boundary faces with a given marker.
    BoundaryIntegral,

    /// Boundary integral divided by boundary measure (area/length).
    BoundaryAverage,

    /// Quadrature-weighted integral over all domain cells.
    DomainIntegral,

    /// Domain integral divided by domain measure (volume/area).
    DomainAverage,

    /// Integral over a region-restricted subset of cells.
    RegionIntegral,

    /// Region integral divided by region measure.
    RegionAverage,

    /// Generic FE expression evaluated per entity (cell, node, QP).
    FEExpression,

    /// Algebraic expression over other auxiliary inputs (no FE field access).
    DerivedCallback,

    /// Nodal sum over boundary face vertices (legacy, non-quadrature-weighted).
    BoundaryNodalSum,
};

// ============================================================================
//  FE quantity shape
// ============================================================================

/**
 * @brief Shape of the FE-backed quantity result.
 */
enum class FEQuantityShapeKind : std::uint8_t {
    Scalar,
    Vector,
    Tensor,
};

/**
 * @brief Shape metadata for an FE-backed quantity.
 */
struct FEQuantityShape {
    FEQuantityShapeKind kind{FEQuantityShapeKind::Scalar};
    int components{1};        ///< Total component count (1 for scalar, dim for vector, dim*dim for tensor).
    int spatial_dim{0};       ///< Spatial dimension (0 = unknown/not applicable).

    [[nodiscard]] static FEQuantityShape scalar() { return {FEQuantityShapeKind::Scalar, 1, 0}; }
    [[nodiscard]] static FEQuantityShape vector(int dim) { return {FEQuantityShapeKind::Vector, dim, dim}; }
    [[nodiscard]] static FEQuantityShape tensor(int dim) { return {FEQuantityShapeKind::Tensor, dim * dim, dim}; }
};

// ============================================================================
//  FE quantity capabilities
// ============================================================================

/**
 * @brief Capability flags for an FE-backed quantity.
 */
struct FEQuantityCapabilities {
    bool explicit_evaluation{true};     ///< Can be evaluated numerically for partitioned coupling.
    bool monolithic_linearization{false}; ///< Can provide dI/du for exact monolithic coupling.
};

// ============================================================================
//  FE quantity definition
// ============================================================================

/**
 * @brief Complete definition of one FE-backed auxiliary input quantity.
 *
 * Stored in `FEQuantityRegistry` and referenced by `AuxiliaryInputHandle`.
 * The definition carries all metadata needed for explicit evaluation,
 * shape-aware binding, and (eventually) monolithic linearization.
 */
struct FEQuantityDefinition {
    /// Unique name (matches the AuxiliaryInputRegistry entry name).
    std::string name{};

    /// What kind of FE quantity this is.
    FEQuantityKind kind{FEQuantityKind::DerivedCallback};

    /// Shape of the result.
    FEQuantityShape shape{};

    /// FE fields referenced by the integrand/expression.
    /// Empty for DerivedCallback and other non-FE-backed quantities.
    std::vector<FieldId> referenced_fields{};

    /// The integrand/expression (for integral/expression kinds).
    forms::FormExpr expression{};

    /// Boundary marker (for BoundaryIntegral, BoundaryAverage, BoundaryNodalSum).
    int boundary_marker{-1};

    /// Region descriptor (for RegionIntegral, RegionAverage).
    /// Uses the same region kind/identity scheme as AuxiliaryDeploymentRegion.
    int region_marker{-1};

    /// Source field name (for SampledField, BoundaryNodalSum).
    std::string source_field_name{};

    /// Entity count (for entity-local quantities like SampledField).
    std::size_t entity_count{0};

    /// Capability flags.
    FEQuantityCapabilities capabilities{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FE_QUANTITY_DEFINITION_H
