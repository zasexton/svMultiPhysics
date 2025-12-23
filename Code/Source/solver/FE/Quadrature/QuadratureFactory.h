/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_QUADRATUREFACTORY_H
#define SVMP_FE_QUADRATURE_QUADRATUREFACTORY_H

/**
 * @file QuadratureFactory.h
 * @brief Factory functions for creating quadrature rules
 *
 * This factory supports two paradigms for reduced integration:
 *
 * 1. Order-based (standard): Uses fewer quadrature points at optimal Gauss
 *    locations. This is the traditional approach used in most FEM codes.
 *
 * 2. Position-based (legacy compatible): Uses the same number of points but
 *    moves them to different locations. This provides compatibility with the
 *    legacy svMultiPhysics qmTET4/qmTRI3 parameters and enables fine-grained
 *    tuning for physics-specific locking mitigation.
 */

#include "QuadratureRule.h"
#include "GaussQuadrature.h"
#include "GaussLobattoQuadrature.h"
#include "TriangleQuadrature.h"
#include "TetrahedronQuadrature.h"
#include "QuadrilateralQuadrature.h"
#include "HexahedronQuadrature.h"
#include "WedgeQuadrature.h"
#include "PyramidQuadrature.h"
#include "PositionBasedParams.h"
#include "PositionBasedTriangleQuadrature.h"
#include "PositionBasedTetrahedronQuadrature.h"
#include "QuadratureCache.h"
#include <memory>

namespace svmp {
namespace FE {
namespace quadrature {

/**
 * @brief Factory for creating quadrature rules
 *
 * Provides static methods for creating both order-based and position-based
 * quadrature rules with optional caching for performance.
 */
class QuadratureFactory {
public:
    // =========================================================================
    // Order-based quadrature (standard approach)
    // =========================================================================

    /**
     * @brief Create an order-based quadrature rule
     *
     * @param element_type The element type (higher-order elements are mapped to their linear base)
     * @param order Desired polynomial exactness
     * @param type Quadrature rule type (GaussLegendre, GaussLobatto, etc.)
     * @param use_cache Whether to cache the rule for reuse
     * @return Shared pointer to the quadrature rule
     * @throws FEException if element type is unsupported or order is invalid
     */
    static std::shared_ptr<const QuadratureRule> create(
        ElementType element_type,
        int order,
        QuadratureType type = QuadratureType::GaussLegendre,
        bool use_cache = true);

    /**
     * @brief Get recommended quadrature order for a given basis order
     *
     * @param basis_order Polynomial order of the basis functions
     * @param is_mass_matrix True for mass matrix assembly (uses higher order)
     * @return Recommended quadrature order
     */
    static int recommended_order(int basis_order, bool is_mass_matrix = false);

    // =========================================================================
    // Position-based quadrature (legacy compatible approach)
    // =========================================================================

    /**
     * @brief Create a position-based quadrature rule
     *
     * Position-based quadrature maintains the same number of points as standard
     * Gaussian quadrature but allows moving points to different locations.
     * This is useful for physics-specific tuning and legacy compatibility.
     *
     * @param element_type Must be Triangle3 or Tetra4 (or their higher-order variants)
     * @param params Position parameters containing the modifier value
     * @param use_cache Whether to cache the rule for reuse
     * @return Shared pointer to the quadrature rule
     * @throws FEException if element type doesn't support position-based quadrature
     */
    static std::shared_ptr<const QuadratureRule> create_position_based(
        ElementType element_type,
        const PositionBasedParams& params,
        bool use_cache = true);

    /**
     * @brief Create a position-based rule from legacy modifier value
     *
     * This provides direct compatibility with the legacy svMultiPhysics solver's
     * qmTET4 and qmTRI3 parameters. The element type determines which parameter
     * interpretation is used.
     *
     * @param element_type Must be Triangle3 or Tetra4 (or their higher-order variants)
     * @param legacy_modifier The legacy qmTET4 or qmTRI3 value
     * @param use_cache Whether to cache the rule for reuse
     * @return Shared pointer to the quadrature rule
     * @throws FEException if element type doesn't support position-based quadrature
     *
     * Example:
     * @code
     *   // Using legacy qmTET4 value from existing solver
     *   auto rule = QuadratureFactory::create_legacy_compatible(
     *       ElementType::Tetra4, mesh.qmTET4);
     * @endcode
     */
    static std::shared_ptr<const QuadratureRule> create_legacy_compatible(
        ElementType element_type,
        Real legacy_modifier,
        bool use_cache = true);

    // =========================================================================
    // Convenience methods for common position-based configurations
    // =========================================================================

    /**
     * @brief Create a central quadrature rule (all points at centroid)
     *
     * For TET4: All 4 points at centroid
     * For TRI3: All 3 points at centroid
     *
     * @param element_type Must be Triangle3 or Tetra4 (or higher-order variants)
     * @param use_cache Whether to cache the rule for reuse
     * @return Shared pointer to the quadrature rule
     */
    static std::shared_ptr<const QuadratureRule> create_central(
        ElementType element_type,
        bool use_cache = true);

    /**
     * @brief Create a nodal quadrature rule (points at vertices)
     *
     * For TET4: 4 points at the 4 vertices
     * For TRI3: 3 points at the 3 vertices
     *
     * @param element_type Must be Triangle3 or Tetra4 (or higher-order variants)
     * @param use_cache Whether to cache the rule for reuse
     * @return Shared pointer to the quadrature rule
     */
    static std::shared_ptr<const QuadratureRule> create_nodal(
        ElementType element_type,
        bool use_cache = true);

    // =========================================================================
    // Query methods
    // =========================================================================

    /**
     * @brief Check if an element type supports position-based quadrature
     *
     * @param element_type The element type to check
     * @return true if position-based quadrature is supported
     */
    static bool supports_position_based(ElementType element_type) noexcept;

    /**
     * @brief Get the default legacy modifier for an element type
     *
     * @param element_type Must be Triangle3 or Tetra4
     * @return Default modifier (Gaussian position)
     * @throws FEException if element type doesn't support position-based quadrature
     */
    static Real default_legacy_modifier(ElementType element_type);
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_QUADRATUREFACTORY_H
