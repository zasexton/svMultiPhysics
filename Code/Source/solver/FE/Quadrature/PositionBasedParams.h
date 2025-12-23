/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_POSITIONBASEDPARAMS_H
#define SVMP_FE_QUADRATURE_POSITIONBASEDPARAMS_H

/**
 * @file PositionBasedParams.h
 * @brief Parameter structure for position-based reduced integration
 *
 * Position-based reduced integration allows fine-grained control over
 * quadrature point placement for simplicial elements (triangles, tetrahedra).
 * This provides compatibility with the legacy svMultiPhysics solver's
 * qmTRI3 and qmTET4 parameters while enabling physics-specific tuning
 * for locking mitigation.
 *
 * Unlike order-based reduced integration (which uses fewer points),
 * position-based integration maintains the same number of points but
 * moves them to different locations within the reference element.
 */

#include "Core/Types.h"
#include <cmath>

namespace svmp {
namespace FE {
namespace quadrature {

/**
 * @brief Parameters for position-based quadrature rules
 *
 * The modifier parameter has element-specific interpretation:
 *
 * TET4 (4-node tetrahedron):
 *   - modifier = s, where points are at (s,t,t), (t,s,t), (t,t,s), (t,t,t)
 *   - t = (1-s)/3
 *   - Valid range: [0.25, 1.0]
 *   - s = 0.25: central rule (all points at centroid)
 *   - s ≈ 0.585: standard Gaussian (default)
 *   - s = 1.0: nodal rule (points at vertices)
 *
 * TRI3 (3-node triangle):
 *   - modifier = s, where points are at (s,t), (t,s), (t,t)
 *   - t = -0.5*s + 0.5
 *   - Valid range: [1/3, 1.0]
 *   - s = 1/3: central rule (all points at centroid)
 *   - s = 2/3: standard Gaussian (default)
 *   - s = 1.0: nodal rule (points at vertices)
 */
struct PositionBasedParams {
    Real modifier = 0.0;  ///< Position parameter (element-specific interpretation)

    /// Default constructor (uses Gaussian position when applied)
    constexpr PositionBasedParams() noexcept = default;

    /// Explicit constructor with modifier value
    constexpr explicit PositionBasedParams(Real mod) noexcept : modifier(mod) {}

    // =========================================================================
    // Named constructors for TET4
    // =========================================================================

    /**
     * @brief Central quadrature rule for TET4
     * All 4 points collapse to the centroid (s = 0.25)
     */
    static constexpr PositionBasedParams tet4_central() noexcept {
        return PositionBasedParams{0.25};
    }

    /**
     * @brief Standard Gaussian quadrature for TET4
     * Optimal 4-point rule with polynomial exactness degree 2
     * s = (5 + 3*sqrt(5))/20 ≈ 0.5854101966249685
     */
    static PositionBasedParams tet4_gaussian() noexcept {
        return PositionBasedParams{(5.0 + 3.0 * std::sqrt(5.0)) / 20.0};
    }

    /**
     * @brief Nodal quadrature rule for TET4
     * Points at the 4 vertices (s = 1.0)
     */
    static constexpr PositionBasedParams tet4_nodal() noexcept {
        return PositionBasedParams{1.0};
    }

    /**
     * @brief Custom TET4 quadrature with user-specified modifier
     * @param s Position parameter in [0.25, 1.0]
     */
    static constexpr PositionBasedParams tet4_custom(Real s) noexcept {
        return PositionBasedParams{s};
    }

    // =========================================================================
    // Named constructors for TRI3
    // =========================================================================

    /**
     * @brief Central quadrature rule for TRI3
     * All 3 points collapse to the centroid (s = 1/3)
     */
    static constexpr PositionBasedParams tri3_central() noexcept {
        return PositionBasedParams{1.0 / 3.0};
    }

    /**
     * @brief Standard Gaussian quadrature for TRI3
     * Optimal 3-point rule with polynomial exactness degree 2
     * s = 2/3 ≈ 0.6666666666666666
     */
    static constexpr PositionBasedParams tri3_gaussian() noexcept {
        return PositionBasedParams{2.0 / 3.0};
    }

    /**
     * @brief Nodal quadrature rule for TRI3
     * Points at the 3 vertices (s = 1.0)
     */
    static constexpr PositionBasedParams tri3_nodal() noexcept {
        return PositionBasedParams{1.0};
    }

    /**
     * @brief Custom TRI3 quadrature with user-specified modifier
     * @param s Position parameter in [1/3, 1.0]
     */
    static constexpr PositionBasedParams tri3_custom(Real s) noexcept {
        return PositionBasedParams{s};
    }

    // =========================================================================
    // Legacy compatibility helpers
    // =========================================================================

    /**
     * @brief Create from legacy qmTET4 parameter value
     *
     * Direct passthrough since qmTET4 uses the same parameterization.
     *
     * @param qm Legacy quadrature modifier value (default: (5+3*sqrt(5))/20)
     */
    static constexpr PositionBasedParams from_qmTET4(Real qm) noexcept {
        return PositionBasedParams{qm};
    }

    /**
     * @brief Create from legacy qmTRI3 parameter value
     *
     * Direct passthrough since qmTRI3 uses the same parameterization.
     *
     * @param qm Legacy quadrature modifier value (default: 2/3)
     */
    static constexpr PositionBasedParams from_qmTRI3(Real qm) noexcept {
        return PositionBasedParams{qm};
    }

    // =========================================================================
    // Validation helpers
    // =========================================================================

    /**
     * @brief Check if modifier is valid for TET4
     * @return true if modifier is in [0.25, 1.0]
     */
    constexpr bool is_valid_for_tet4() const noexcept {
        return modifier >= 0.25 && modifier <= 1.0;
    }

    /**
     * @brief Check if modifier is valid for TRI3
     * @return true if modifier is in [1/3, 1.0]
     */
    bool is_valid_for_tri3() const noexcept {
        return modifier >= (1.0 / 3.0 - 1e-14) && modifier <= 1.0;
    }

    // =========================================================================
    // Comparison operators
    // =========================================================================

    constexpr bool operator==(const PositionBasedParams& other) const noexcept {
        // Use tolerance for floating-point comparison
        return std::abs(modifier - other.modifier) < 1e-14;
    }

    constexpr bool operator!=(const PositionBasedParams& other) const noexcept {
        return !(*this == other);
    }
};

/**
 * @brief Default position-based parameters
 *
 * These constants provide quick access to the most common configurations.
 */
namespace position_defaults {

/// Default Gaussian modifier for TET4: (5 + 3*sqrt(5))/20
inline const Real TET4_GAUSSIAN = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;

/// Central modifier for TET4: 0.25
constexpr Real TET4_CENTRAL = 0.25;

/// Nodal modifier for TET4: 1.0
constexpr Real TET4_NODAL = 1.0;

/// Default Gaussian modifier for TRI3: 2/3
constexpr Real TRI3_GAUSSIAN = 2.0 / 3.0;

/// Central modifier for TRI3: 1/3
constexpr Real TRI3_CENTRAL = 1.0 / 3.0;

/// Nodal modifier for TRI3: 1.0
constexpr Real TRI3_NODAL = 1.0;

} // namespace position_defaults

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_POSITIONBASEDPARAMS_H
