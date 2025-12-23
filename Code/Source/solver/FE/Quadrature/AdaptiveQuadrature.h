/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_ADAPTIVEQUADRATURE_H
#define SVMP_FE_QUADRATURE_ADAPTIVEQUADRATURE_H

/**
 * @file AdaptiveQuadrature.h
 * @brief Error-driven adaptive numerical integration on reference elements
 */

#include "QuadratureRule.h"
#include "CompositeQuadrature.h"
#include <functional>

namespace svmp {
namespace FE {
namespace quadrature {

class AdaptiveQuadrature {
public:
    struct Result {
        Real value = 0;
        Real estimate = 0;
        bool converged = false;
        int levels_used = 0;
    };

    AdaptiveQuadrature(Real tolerance = 1e-8, int max_levels = 4)
        : tolerance_(tolerance), max_levels_(max_levels) {}

    /**
     * @brief Integrate a scalar function over the reference element using adaptive refinement
     * @param rule Base quadrature rule
     * @param f Integrand f(ξ)
     */
    Result integrate(const QuadratureRule& rule,
                     const std::function<Real(const QuadPoint&)>& f) const;

private:
    Real tolerance_;
    int max_levels_;

    static Real evaluate(const QuadratureRule& rule,
                         const std::function<Real(const QuadPoint&)>& f);
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_ADAPTIVEQUADRATURE_H
