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
 * elements (line/quad/hex) are fully supported; triangles and wedges are
 * subdivided using barycentric grids. Other element families fallback to the
 * provided rule without subdivision to preserve correctness.
 */
class CompositeQuadrature : public QuadratureRule {
public:
    explicit CompositeQuadrature(const QuadratureRule& base_rule,
                                 int subdivisions);

    CompositeQuadrature(const QuadratureRule& base_rule,
                        const std::array<int, 3>& subdivisions);
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_COMPOSITEQUADRATURE_H
