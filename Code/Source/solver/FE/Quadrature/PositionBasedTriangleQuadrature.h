/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_POSITIONBASEDTRIANGLEQUADRATURE_H
#define SVMP_FE_QUADRATURE_POSITIONBASEDTRIANGLEQUADRATURE_H

/**
 * @file PositionBasedTriangleQuadrature.h
 * @brief Position-based quadrature for 3-node triangles
 *
 * Implements a 3-point quadrature rule for triangles where the point
 * positions are controlled by a single parameter. This provides compatibility
 * with the legacy svMultiPhysics qmTRI3 parameter and enables fine-grained
 * control over integration behavior for locking mitigation.
 *
 * Reference element: Triangle with vertices at (0,0), (1,0), (0,1)
 * Reference measure: 0.5
 *
 * Point positions:
 *   - Point 0: (t, t)
 *   - Point 1: (s, t)
 *   - Point 2: (t, s)
 *   where s is the modifier parameter and t = -0.5*s + 0.5
 *
 * Special cases:
 *   - s = 1/3: Central rule (all points at centroid)
 *   - s = 2/3: Standard 3-point Gaussian (default legacy behavior)
 *   - s = 1.0: Nodal rule (points at vertices)
 */

#include "QuadratureRule.h"
#include "PositionBasedParams.h"

namespace svmp {
namespace FE {
namespace quadrature {

/**
 * @brief Position-based 3-point quadrature rule for triangles
 *
 * This rule always uses exactly 3 quadrature points with equal weights
 * of 1/6 each (summing to 0.5, the reference triangle measure). The
 * position parameter controls where these points are located within
 * the reference triangle.
 */
class PositionBasedTriangleQuadrature : public QuadratureRule {
public:
    /**
     * @brief Construct a position-based triangle quadrature rule
     *
     * @param params Position parameters containing the modifier value
     * @throws FEException if modifier is outside valid range [1/3, 1.0]
     */
    explicit PositionBasedTriangleQuadrature(const PositionBasedParams& params);

    /**
     * @brief Construct with explicit modifier value
     *
     * @param modifier Position parameter s in [1/3, 1.0]
     * @throws FEException if modifier is outside valid range
     */
    explicit PositionBasedTriangleQuadrature(Real modifier);

    /**
     * @brief Get the position modifier used by this rule
     * @return The modifier value s
     */
    Real modifier() const noexcept { return modifier_; }

    /**
     * @brief Check if this rule uses the standard Gaussian position
     * @return true if modifier is approximately 2/3
     */
    bool is_gaussian() const noexcept;

    /**
     * @brief Check if this rule is the central (single-point effective) rule
     * @return true if modifier is approximately 1/3
     */
    bool is_central() const noexcept;

    /**
     * @brief Check if this rule is the nodal rule
     * @return true if modifier is approximately 1.0
     */
    bool is_nodal() const noexcept;

private:
    Real modifier_;  ///< The position parameter s

    /**
     * @brief Compute the polynomial exactness order based on position
     *
     * The Gaussian position (s = 2/3) achieves order 2.
     * Non-Gaussian positions achieve reduced accuracy.
     *
     * @param s Position modifier
     * @return Estimated polynomial exactness order
     */
    static int compute_order(Real s) noexcept;

    /**
     * @brief Initialize points and weights from modifier
     * @param s Position modifier
     */
    void initialize(Real s);
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_POSITIONBASEDTRIANGLEQUADRATURE_H
