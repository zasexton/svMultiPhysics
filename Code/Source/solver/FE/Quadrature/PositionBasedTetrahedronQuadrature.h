/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_POSITIONBASEDTETRAHEDRONQUADRATURE_H
#define SVMP_FE_QUADRATURE_POSITIONBASEDTETRAHEDRONQUADRATURE_H

/**
 * @file PositionBasedTetrahedronQuadrature.h
 * @brief Position-based quadrature for 4-node tetrahedra
 *
 * Implements a 4-point quadrature rule for tetrahedra where the point
 * positions are controlled by a single parameter. This provides compatibility
 * with the legacy svMultiPhysics qmTET4 parameter and enables fine-grained
 * control over integration behavior for locking mitigation in nearly
 * incompressible materials.
 *
 * Reference element: Tetrahedron with vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
 * Reference measure: 1/6
 *
 * Point positions in barycentric-like coordinates:
 *   - Point 0: (s, t, t)
 *   - Point 1: (t, s, t)
 *   - Point 2: (t, t, s)
 *   - Point 3: (t, t, t)
 *   where s is the modifier parameter and t = (1-s)/3
 *
 * Special cases:
 *   - s = 0.25: Central rule (all points at centroid)
 *   - s = (5+3*sqrt(5))/20 ≈ 0.585: Standard 4-point Gaussian (default legacy behavior)
 *   - s = 1.0: Nodal rule (points at vertices)
 */

#include "QuadratureRule.h"
#include "PositionBasedParams.h"

namespace svmp {
namespace FE {
namespace quadrature {

/**
 * @brief Position-based 4-point quadrature rule for tetrahedra
 *
 * This rule always uses exactly 4 quadrature points with equal weights
 * of 1/24 each (summing to 1/6, the reference tetrahedron volume). The
 * position parameter controls where these points are located within
 * the reference tetrahedron.
 */
class PositionBasedTetrahedronQuadrature : public QuadratureRule {
public:
    /**
     * @brief Construct a position-based tetrahedron quadrature rule
     *
     * @param params Position parameters containing the modifier value
     * @throws FEException if modifier is outside valid range [0.25, 1.0]
     */
    explicit PositionBasedTetrahedronQuadrature(const PositionBasedParams& params);

    /**
     * @brief Construct with explicit modifier value
     *
     * @param modifier Position parameter s in [0.25, 1.0]
     * @throws FEException if modifier is outside valid range
     */
    explicit PositionBasedTetrahedronQuadrature(Real modifier);

    /**
     * @brief Get the position modifier used by this rule
     * @return The modifier value s
     */
    Real modifier() const noexcept { return modifier_; }

    /**
     * @brief Check if this rule uses the standard Gaussian position
     * @return true if modifier is approximately (5+3*sqrt(5))/20
     */
    bool is_gaussian() const noexcept;

    /**
     * @brief Check if this rule is the central (single-point effective) rule
     * @return true if modifier is approximately 0.25
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
     * The Gaussian position (s ≈ 0.585) achieves order 2.
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

#endif // SVMP_FE_QUADRATURE_POSITIONBASEDTETRAHEDRONQUADRATURE_H
