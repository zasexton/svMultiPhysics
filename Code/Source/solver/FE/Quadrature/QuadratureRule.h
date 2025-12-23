/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_RULE_H
#define SVMP_FE_QUADRATURE_RULE_H

/**
 * @file QuadratureRule.h
 * @brief Abstracted quadrature rule representation for FE integration
 *
 * This header defines the base class for all quadrature rules used by the
 * finite element infrastructure. Rules are expressed in reference element
 * space only; mapping to physical space is handled by the Geometry module.
 *
 * The interface is intentionally lightweight and header-only to avoid coupling
 * Quadrature to other modules while remaining compatible with the Mesh library
 * through shared type aliases provided by FE/Core/Types.h.
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Math/Vector.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace quadrature {

/// Convenience alias for quadrature point representation in reference space
using QuadPoint = math::Vector<Real, 3>;

/**
 * @brief Base class for quadrature rules over reference elements
 *
 * Derived classes populate the point/weight data via the protected setters.
 * The class performs lightweight consistency checks (size agreement, basic
 * reference-measure validation) but leaves element-specific checks to callers.
 */
class QuadratureRule {
public:
    virtual ~QuadratureRule() = default;

    /// Number of quadrature points
    std::size_t num_points() const noexcept { return points_.size(); }

    /// Polynomial exactness degree reported by the rule
    int order() const noexcept { return order_; }

    /// Spatial dimension of the reference domain
    int dimension() const noexcept { return dimension_; }

    /// Cell family the rule integrates over (line, tri, quad, ...)
    svmp::CellFamily cell_family() const noexcept { return cell_family_; }

    /// Access a single quadrature point (no bounds checking)
    QuadPoint point(std::size_t i) const noexcept { return points_[i]; }

    /// Access a single quadrature weight (no bounds checking)
    Real weight(std::size_t i) const noexcept { return weights_[i]; }

    /// Bulk accessors
    const std::vector<QuadPoint>& points() const noexcept { return points_; }
    const std::vector<Real>& weights() const noexcept { return weights_; }

    /**
     * @brief Validate rule data for basic consistency
     * @param tol Relative tolerance for weight sum check
     * @return True if rule passes size and weight checks
     */
    virtual bool is_valid(Real tol = 1e-12) const;

    /**
     * @brief Reference-domain measure for the element family
     *
     * Length/area/volume of the canonical reference element:
     * - Line   [-1,1]            -> 2
     * - Quad   [-1,1]^2          -> 4
     * - Hex    [-1,1]^3          -> 8
     * - Tri    (0,0)-(1,0)-(0,1) -> 0.5
     * - Tet    simplex at origin -> 1/6
     * - Wedge  (triangle x line) -> 1
     * - Pyramid (x,y in [-1,1], z in [0,1]) -> 4/3
     */
    Real reference_measure() const noexcept;

protected:
    QuadratureRule(svmp::CellFamily family, int dimension, int order = 0)
        : cell_family_(family), dimension_(dimension), order_(order) {}

    /// Assign point and weight storage (sizes must match)
    void set_data(std::vector<QuadPoint> pts, std::vector<Real> wts);

    /// Override computed order in derived classes
    void set_order(int ord) noexcept { order_ = ord; }

private:
    svmp::CellFamily cell_family_;
    int dimension_;
    int order_;
    std::vector<QuadPoint> points_;
    std::vector<Real> weights_;
};

// --------------------------------------------------------------------------------
// Inline implementations
// --------------------------------------------------------------------------------

inline void QuadratureRule::set_data(std::vector<QuadPoint> pts, std::vector<Real> wts) {
    if (pts.size() != wts.size()) {
        throw FEException("QuadratureRule: points/weights size mismatch",
                          __FILE__, __LINE__, __func__, FEStatus::QuadratureError);
    }
    points_ = std::move(pts);
    weights_ = std::move(wts);
}

inline bool QuadratureRule::is_valid(Real tol) const {
    if (points_.empty() || points_.size() != weights_.size()) {
        return false;
    }
    Real sum_w = Real(0);
    for (Real w : weights_) {
        if (!std::isfinite(w)) {
            return false;
        }
        sum_w += w;
    }
    const Real ref = reference_measure();
    const Real denom = std::max(Real(1), std::abs(ref));
    return std::abs(sum_w - ref) <= tol * denom;
}

inline Real QuadratureRule::reference_measure() const noexcept {
    switch (cell_family_) {
        case svmp::CellFamily::Line:      return Real(2);
        case svmp::CellFamily::Quad:      return Real(4);
        case svmp::CellFamily::Hex:       return Real(8);
        case svmp::CellFamily::Triangle:  return Real(0.5);
        case svmp::CellFamily::Tetra:     return Real(1.0 / 6.0);
        case svmp::CellFamily::Wedge:     return Real(1.0);     // 0.5 area * length 2
        case svmp::CellFamily::Pyramid:   return Real(4.0 / 3.0);
        case svmp::CellFamily::Point:     return Real(1.0);
        default:                          return Real(1.0);
    }
}

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_RULE_H
