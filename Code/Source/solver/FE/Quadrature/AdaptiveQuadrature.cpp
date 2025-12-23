/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "AdaptiveQuadrature.h"
#include <cmath>

namespace svmp {
namespace FE {
namespace quadrature {

Real AdaptiveQuadrature::evaluate(const QuadratureRule& rule,
                                  const std::function<Real(const QuadPoint&)>& f) {
    Real sum = Real(0);
    const auto& pts = rule.points();
    const auto& wts = rule.weights();
    for (std::size_t i = 0; i < pts.size(); ++i) {
        sum += wts[i] * f(pts[i]);
    }
    return sum;
}

AdaptiveQuadrature::Result AdaptiveQuadrature::integrate(
    const QuadratureRule& rule,
    const std::function<Real(const QuadPoint&)>& f) const {

    Result result;
    result.value = evaluate(rule, f);
    result.estimate = result.value;

    Real previous = result.value;

    for (int level = 1; level <= max_levels_; ++level) {
        int subdivisions = 1 << level;  // 2, 4, 8, ...
        const int sub_safe = std::max(2, subdivisions);
        CompositeQuadrature refined(rule, sub_safe);
        Real refined_value = evaluate(refined, f);

        result.levels_used = level;
        result.estimate = refined_value;
        result.value = refined_value;

        Real denom = std::max(std::abs(refined_value), Real(1));
        // Use successive-refinement difference as error estimator
        if (std::abs(refined_value - previous) <= tolerance_ * denom) {
            result.converged = true;
            break;
        }

        previous = refined_value;
    }

    return result;
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
