/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_GEOMETRY_INVERSEMAPPING_H
#define SVMP_FE_GEOMETRY_INVERSEMAPPING_H

/**
 * @file InverseMapping.h
 * @brief Newton-based inverse mapping with optional line search for robustness
 *
 * Provides methods to find reference coordinates xi such that x(xi) = x_phys
 * for a given geometry mapping. The enhanced solve_with_line_search method
 * is recommended for severely distorted elements where standard Newton
 * iteration may diverge.
 */

#include "GeometryMapping.h"

namespace svmp {
namespace FE {
namespace geometry {

/**
 * @brief Configuration options for inverse mapping solver
 */
struct InverseMappingOptions {
    /// Maximum number of Newton iterations
    int max_iters = 25;

    /// Convergence tolerance for residual norm
    Real tol = Real(1e-12);

    /// Absolute determinant threshold for singular Jacobians
#ifndef FE_INVERSEMAPPING_SINGULAR_TOL
    Real singular_tol = Real(1e-14);
#else
    Real singular_tol = Real(FE_INVERSEMAPPING_SINGULAR_TOL);
#endif

    /// Enable backtracking line search for robustness
    bool use_line_search = false;

    /// Armijo condition parameter (sufficient decrease)
    Real armijo_c = Real(1e-4);

    /// Line search contraction factor
    Real line_search_rho = Real(0.5);

    /// Maximum line search iterations
    int max_line_search_iters = 10;

    /// Minimum step size before declaring failure
    Real min_step_size = Real(1e-10);

    /// Require converged reference point to lie inside the reference element
    bool require_inside = false;

    /// Tolerance used by inside-element checks when enabled
    Real inside_tol = Real(1e-12);
};

/**
 * @brief Newton-based inverse mapping solver
 *
 * Finds reference coordinates xi such that the mapping x(xi) equals
 * the target physical point x_phys. Supports optional backtracking
 * line search for improved robustness on distorted elements.
 */
class InverseMapping {
public:
    /**
     * @brief Standard Newton solver (legacy interface)
     * @param mapping The geometry mapping to invert
     * @param x_phys Target physical coordinates
     * @param initial_guess Starting point in reference space
     * @param max_iters Maximum Newton iterations
     * @param tol Convergence tolerance
     * @return Reference coordinates xi such that x(xi) = x_phys
     * @throws FEException if convergence fails or Jacobian is singular
     */
    static math::Vector<Real, 3> solve(const GeometryMapping& mapping,
                                       const math::Vector<Real, 3>& x_phys,
                                       const math::Vector<Real, 3>& initial_guess = math::Vector<Real, 3>{},
                                       int max_iters = 25,
                                       Real tol = Real(1e-12));

    /**
     * @brief Enhanced Newton solver with optional line search
     * @param mapping The geometry mapping to invert
     * @param x_phys Target physical coordinates
     * @param initial_guess Starting point in reference space
     * @param opts Solver configuration options
     * @return Reference coordinates xi such that x(xi) = x_phys
     * @throws FEException if convergence fails or Jacobian is singular
     *
     * When opts.use_line_search is true, uses backtracking line search
     * with Armijo condition to ensure sufficient decrease in the residual
     * norm at each iteration. This improves robustness for severely
     * distorted elements where the standard Newton direction may overshoot.
     */
    static math::Vector<Real, 3> solve_with_options(
        const GeometryMapping& mapping,
        const math::Vector<Real, 3>& x_phys,
        const math::Vector<Real, 3>& initial_guess,
        const InverseMappingOptions& opts);

    /**
     * @brief Convenience method with line search enabled
     * @param mapping The geometry mapping to invert
     * @param x_phys Target physical coordinates
     * @param initial_guess Starting point in reference space
     * @param max_iters Maximum Newton iterations
     * @param tol Convergence tolerance
     * @return Reference coordinates xi such that x(xi) = x_phys
     *
     * Equivalent to solve_with_options with use_line_search = true.
     * Recommended for elements with high aspect ratio or skewness.
     */
    static math::Vector<Real, 3> solve_robust(
        const GeometryMapping& mapping,
        const math::Vector<Real, 3>& x_phys,
        const math::Vector<Real, 3>& initial_guess = math::Vector<Real, 3>{},
        int max_iters = 25,
        Real tol = Real(1e-12));

private:
    /**
     * @brief Compute residual norm squared
     */
    static Real residual_norm_sq(const GeometryMapping& mapping,
                                 const math::Vector<Real, 3>& xi,
                                 const math::Vector<Real, 3>& x_phys);
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_INVERSEMAPPING_H
