/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ELEMENTS_ELEMENTVALIDATOR_H
#define SVMP_FE_ELEMENTS_ELEMENTVALIDATOR_H

/**
 * @file ElementValidator.h
 * @brief Quality checks for finite elements using geometry quadrature
 */

#include "Elements/Element.h"
#include "Geometry/GeometryValidator.h"

namespace svmp {
namespace FE {
namespace elements {

/**
 * @brief Aggregated quality metrics for an element over its quadrature points
 *
 * Reports geometric quality indicators sampled at all quadrature points
 * of the element. These metrics are useful for detecting degenerate or
 * poorly-conditioned elements that may cause numerical issues during assembly.
 */
struct ElementQuality {
    /// True if all sampled Jacobian determinants are positive (element not inverted)
    bool positive_jacobian{true};

    /// Minimum Jacobian determinant across all quadrature points.
    /// Values <= 0 indicate element inversion; very small positive values
    /// indicate near-degenerate elements.
    Real min_detJ{std::numeric_limits<Real>::max()};

    /**
     * @brief Maximum condition number of the Jacobian across all quadrature points
     *
     * The condition number κ = ||J|| · ||J⁻¹|| measures how well-conditioned
     * the element mapping is. Interpretation:
     * - κ ≈ 1: Nearly optimal element shape (e.g., equilateral triangle, unit square)
     * - κ < 10: Good quality element, typically acceptable for most simulations
     * - κ > 100: Poor quality element, may cause numerical instability or slow convergence
     * - κ → ∞: Degenerate element with singular Jacobian (inverted or collapsed)
     *
     * High condition numbers indicate:
     * - Highly skewed or elongated elements (aspect ratio issues)
     * - Elements approaching degeneracy (e.g., near-zero angles)
     * - Potential accuracy loss in gradient computations
     * - Increased sensitivity to round-off errors
     *
     * For adaptive mesh refinement, elements with κ > threshold should be
     * flagged for coarsening or smoothing.
     */
    Real max_condition_number{Real(0)};
};

class ElementValidator {
public:
    /**
     * @brief Evaluate geometric quality of an element mapping
     *
     * The mapping is sampled at all quadrature points provided by the element.
     * The minimum Jacobian determinant and the maximum condition number of the
     * Jacobian are reported, along with a boolean indicating whether all
     * sampled detJ values are positive.
     */
    static ElementQuality validate(const Element& element,
                                   const geometry::GeometryMapping& mapping);
};

} // namespace elements
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ELEMENTS_ELEMENTVALIDATOR_H

