/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_COMPOSITEQUADRATURE_H
#define SVMP_FE_QUADRATURE_COMPOSITEQUADRATURE_H

/**
 * @file CompositeQuadrature.h
 * @brief Subdivision-based quadrature wrappers
 */

#include "QuadratureRule.h"
#include <array>

namespace svmp {
namespace FE {
namespace quadrature {

/**
 * @brief Build composite rules by uniformly subdividing reference elements
 *
 * CompositeQuadrature reuses an existing rule on the reference element and
 * applies it to each subcell, scaling points/weights appropriately. Tensor
 * elements use Cartesian subdivisions. Triangle and tetrahedron grids use
 * the first subdivision count on all simplex axes. Wedges also refine the
 * axial direction. Pyramid layers use collapsed Gauss quadrature, retaining
 * the axis orders of a PyramidQuadrature or the declared degree of another
 * pyramid rule. A subdivision count of one preserves the supplied rule.
 */
class CompositeQuadrature : public QuadratureRule {
public:
    explicit CompositeQuadrature(const QuadratureRule& base_rule,
                                 int subdivisions);

    CompositeQuadrature(const QuadratureRule& base_rule,
                        const std::array<int, 3>& subdivisions);
};

/// Apply a tensor-cell rule on each box of an explicit reference partition.
class TensorPartitionQuadrature : public QuadratureRule {
public:
    using AxisBreaks = std::vector<std::vector<Real>>;

    /// Each active axis lists its increasing endpoints, including -1 and +1.
    TensorPartitionQuadrature(const QuadratureRule& base_rule, AxisBreaks axis_breaks);
    const AxisBreaks& axis_breaks() const noexcept { return axis_breaks_; }

private:
    AxisBreaks axis_breaks_;
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_COMPOSITEQUADRATURE_H
