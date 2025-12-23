/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "InverseMapping.h"
#include <cmath>
#include <limits>
#include <string>

namespace svmp {
namespace FE {
namespace geometry {

namespace {

const char* element_type_name(ElementType type) {
    switch (type) {
        case ElementType::Point1:    return "Point1";
        case ElementType::Line2:     return "Line2";
        case ElementType::Line3:     return "Line3";
        case ElementType::Triangle3: return "Triangle3";
        case ElementType::Triangle6: return "Triangle6";
        case ElementType::Quad4:     return "Quad4";
        case ElementType::Quad8:     return "Quad8";
        case ElementType::Quad9:     return "Quad9";
        case ElementType::Tetra4:    return "Tetra4";
        case ElementType::Tetra10:   return "Tetra10";
        case ElementType::Hex8:      return "Hex8";
        case ElementType::Hex20:     return "Hex20";
        case ElementType::Hex27:     return "Hex27";
        case ElementType::Wedge6:    return "Wedge6";
        case ElementType::Wedge15:   return "Wedge15";
        case ElementType::Wedge18:   return "Wedge18";
        case ElementType::Pyramid5:  return "Pyramid5";
        case ElementType::Pyramid13: return "Pyramid13";
        case ElementType::Pyramid14: return "Pyramid14";
        default:                     return "Unknown";
    }
}

bool is_inside_reference_element(ElementType type, const math::Vector<Real, 3>& xi, Real tol) {
    switch (type) {
        case ElementType::Point1:
            return true;

        case ElementType::Line2:
        case ElementType::Line3:
            return xi[0] >= Real(-1) - tol && xi[0] <= Real(1) + tol;

        case ElementType::Quad4:
        case ElementType::Quad8:
        case ElementType::Quad9:
            return xi[0] >= Real(-1) - tol && xi[0] <= Real(1) + tol &&
                   xi[1] >= Real(-1) - tol && xi[1] <= Real(1) + tol;

        case ElementType::Hex8:
        case ElementType::Hex20:
        case ElementType::Hex27:
            return xi[0] >= Real(-1) - tol && xi[0] <= Real(1) + tol &&
                   xi[1] >= Real(-1) - tol && xi[1] <= Real(1) + tol &&
                   xi[2] >= Real(-1) - tol && xi[2] <= Real(1) + tol;

        case ElementType::Triangle3:
        case ElementType::Triangle6:
            return xi[0] >= Real(0) - tol && xi[1] >= Real(0) - tol &&
                   (xi[0] + xi[1]) <= Real(1) + tol;

        case ElementType::Tetra4:
        case ElementType::Tetra10:
            return xi[0] >= Real(0) - tol && xi[1] >= Real(0) - tol && xi[2] >= Real(0) - tol &&
                   (xi[0] + xi[1] + xi[2]) <= Real(1) + tol;

        case ElementType::Wedge6:
        case ElementType::Wedge15:
        case ElementType::Wedge18:
            return xi[0] >= Real(0) - tol && xi[1] >= Real(0) - tol &&
                   (xi[0] + xi[1]) <= Real(1) + tol &&
                   xi[2] >= Real(-1) - tol && xi[2] <= Real(1) + tol;

        case ElementType::Pyramid5:
        case ElementType::Pyramid13:
        case ElementType::Pyramid14: {
            const Real z = xi[2];
            if (z < Real(0) - tol || z > Real(1) + tol) {
                return false;
            }
            const Real r = Real(1) - z;
            return std::abs(xi[0]) <= r + tol && std::abs(xi[1]) <= r + tol;
        }

        default:
            return false;
    }
}

math::Vector<Real, 3> default_initial_guess(ElementType type) {
    switch (type) {
        case ElementType::Triangle3:
        case ElementType::Triangle6:
            return math::Vector<Real, 3>{Real(1.0/3.0), Real(1.0/3.0), Real(0)};
        case ElementType::Tetra4:
        case ElementType::Tetra10:
            return math::Vector<Real, 3>{Real(0.25), Real(0.25), Real(0.25)};
        case ElementType::Wedge6:
        case ElementType::Wedge15:
        case ElementType::Wedge18:
            return math::Vector<Real, 3>{Real(1.0/3.0), Real(1.0/3.0), Real(0)};
        case ElementType::Pyramid5:
        case ElementType::Pyramid13:
        case ElementType::Pyramid14:
            return math::Vector<Real, 3>{Real(0), Real(0), Real(0.25)};
        default:
            return math::Vector<Real, 3>{};
    }
}

} // namespace

Real InverseMapping::residual_norm_sq(const GeometryMapping& mapping,
                                      const math::Vector<Real, 3>& xi,
                                      const math::Vector<Real, 3>& x_phys) {
    auto x_map = mapping.map_to_physical(xi);
    Real norm_sq = Real(0);
    for (std::size_t i = 0; i < 3; ++i) {
        const Real diff = x_map[i] - x_phys[i];
        norm_sq += diff * diff;
    }
    return norm_sq;
}

math::Vector<Real, 3> InverseMapping::solve(const GeometryMapping& mapping,
                                           const math::Vector<Real, 3>& x_phys,
                                           const math::Vector<Real, 3>& initial_guess,
                                           int max_iters,
                                           Real tol) {
    InverseMappingOptions opts;
    opts.max_iters = max_iters;
    opts.tol = tol;
    opts.use_line_search = false;
    return solve_with_options(mapping, x_phys, initial_guess, opts);
}

math::Vector<Real, 3> InverseMapping::solve_robust(const GeometryMapping& mapping,
                                                   const math::Vector<Real, 3>& x_phys,
                                                   const math::Vector<Real, 3>& initial_guess,
                                                   int max_iters,
                                                   Real tol) {
    InverseMappingOptions opts;
    opts.max_iters = max_iters;
    opts.tol = tol;
    opts.use_line_search = true;
    return solve_with_options(mapping, x_phys, initial_guess, opts);
}

math::Vector<Real, 3> InverseMapping::solve_with_options(
    const GeometryMapping& mapping,
    const math::Vector<Real, 3>& x_phys,
    const math::Vector<Real, 3>& initial_guess,
    const InverseMappingOptions& opts) {

    math::Vector<Real, 3> xi = initial_guess;
    // Default initial guess at element centroid if zero vector provided.
    if (xi[0] == Real(0) && xi[1] == Real(0) && xi[2] == Real(0)) {
        xi = default_initial_guess(mapping.element_type());
    }

    const int dim = mapping.dimension();
    const std::size_t sdim = static_cast<std::size_t>(dim);
    const Real tol_sq = opts.tol * opts.tol;
    Real last_norm_sq = std::numeric_limits<Real>::infinity();

    for (int iter = 0; iter < opts.max_iters; ++iter) {
        // Compute residual f(xi) = x(xi) - x_phys
        auto x_map = mapping.map_to_physical(xi);
        math::Vector<Real, 3> residual{};
        Real norm_sq = Real(0);
        for (std::size_t i = 0; i < 3; ++i) {
            residual[i] = x_map[i] - x_phys[i];
            norm_sq += residual[i] * residual[i];
        }
        last_norm_sq = norm_sq;

        // Check convergence
        if (norm_sq < tol_sq) {
            if (opts.require_inside && !is_inside_reference_element(mapping.element_type(), xi, opts.inside_tol)) {
                throw FEException("InverseMapping converged outside reference element for type " +
                                      std::string(element_type_name(mapping.element_type())),
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            return xi;
        }

        // Check Jacobian singularity
        Real det = mapping.jacobian_determinant(xi);
        if (std::abs(det) < opts.singular_tol) {
            throw FEException("InverseMapping encountered singular Jacobian for type " +
                                  std::string(element_type_name(mapping.element_type())),
                              __FILE__, __LINE__, __func__, FEStatus::SingularMapping);
        }

        // Compute Newton direction by decomposing the residual in the local Jacobian frame:
        // delta = J^{-1} * residual, then update reference coordinates with delta[0..dim-1].
        const auto Jinv = mapping.jacobian_inverse(xi);
        math::Vector<Real, 3> delta{};
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                delta[i] += Jinv(i, j) * residual[j];
            }
        }

        // Apply update with optional line search
        if (opts.use_line_search) {
            // Backtracking line search with Armijo condition
            Real alpha = Real(1);
            const Real current_norm_sq = norm_sq;
            bool accepted = false;

            for (int ls_iter = 0; ls_iter < opts.max_line_search_iters; ++ls_iter) {
                // Try step
                math::Vector<Real, 3> xi_new = xi;
                for (std::size_t i = 0; i < sdim; ++i) {
                    xi_new[i] = xi[i] - alpha * delta[i];
                }

                const Real new_norm_sq = residual_norm_sq(mapping, xi_new, x_phys);

                // Armijo sufficient decrease condition
                if (new_norm_sq <= (Real(1) - opts.armijo_c * alpha) * current_norm_sq) {
                    xi = xi_new;
                    accepted = true;
                    break;
                }

                // Contract step size
                alpha *= opts.line_search_rho;

                if (alpha < opts.min_step_size) {
                    break;
                }
            }

            if (!accepted) {
                // Line search failed: accept full step anyway (may still converge)
                for (std::size_t i = 0; i < sdim; ++i) {
                    xi[i] -= delta[i];
                }
            }
        } else {
            // Standard Newton update
            for (std::size_t i = 0; i < sdim; ++i) {
                xi[i] -= delta[i];
            }
        }

        // Check step size convergence
        Real delta_norm_sq = Real(0);
        for (std::size_t i = 0; i < sdim; ++i) {
            delta_norm_sq += delta[i] * delta[i];
        }
        if (delta_norm_sq < tol_sq) {
            if (opts.require_inside && !is_inside_reference_element(mapping.element_type(), xi, opts.inside_tol)) {
                throw FEException("InverseMapping converged outside reference element for type " +
                                      std::string(element_type_name(mapping.element_type())),
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            return xi;
        }
    }

    throw FEException("InverseMapping failed to converge for type " +
                          std::string(element_type_name(mapping.element_type())) +
                          " (||r||=" + std::to_string(std::sqrt(last_norm_sq)) + ")",
                      __FILE__, __LINE__, __func__, FEStatus::ConvergenceError);
}

} // namespace geometry
} // namespace FE
} // namespace svmp
