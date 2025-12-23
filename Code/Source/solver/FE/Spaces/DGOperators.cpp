/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/DGOperators.h"
#include "Core/FEException.h"
#include <cmath>

namespace svmp {
namespace FE {
namespace spaces {

// =============================================================================
// Scalar Jump and Average
// =============================================================================

std::vector<Real> DGOperators::jump(
    const std::vector<Real>& values_plus,
    const std::vector<Real>& values_minus) {

    FE_CHECK_ARG(values_plus.size() == values_minus.size(),
        "Plus and minus arrays must have same size");

    std::vector<Real> result(values_plus.size());
    for (std::size_t i = 0; i < values_plus.size(); ++i) {
        result[i] = values_plus[i] - values_minus[i];
    }
    return result;
}

std::vector<Real> DGOperators::average(
    const std::vector<Real>& values_plus,
    const std::vector<Real>& values_minus) {

    FE_CHECK_ARG(values_plus.size() == values_minus.size(),
        "Plus and minus arrays must have same size");

    std::vector<Real> result(values_plus.size());
    for (std::size_t i = 0; i < values_plus.size(); ++i) {
        result[i] = Real(0.5) * (values_plus[i] + values_minus[i]);
    }
    return result;
}

std::vector<Real> DGOperators::weighted_average(
    const std::vector<Real>& values_plus,
    const std::vector<Real>& values_minus,
    Real weight_plus,
    Real weight_minus) {

    FE_CHECK_ARG(values_plus.size() == values_minus.size(),
        "Plus and minus arrays must have same size");

    std::vector<Real> result(values_plus.size());
    for (std::size_t i = 0; i < values_plus.size(); ++i) {
        result[i] = weight_plus * values_plus[i] + weight_minus * values_minus[i];
    }
    return result;
}

// =============================================================================
// 3D Vector Jump and Average
// =============================================================================

std::vector<DGOperators::Vec3> DGOperators::jump_vector(
    const std::vector<Vec3>& values_plus,
    const std::vector<Vec3>& values_minus) {

    FE_CHECK_ARG(values_plus.size() == values_minus.size(),
        "Plus and minus arrays must have same size");

    std::vector<Vec3> result(values_plus.size());
    for (std::size_t i = 0; i < values_plus.size(); ++i) {
        result[i] = values_plus[i] - values_minus[i];
    }
    return result;
}

std::vector<DGOperators::Vec3> DGOperators::average_vector(
    const std::vector<Vec3>& values_plus,
    const std::vector<Vec3>& values_minus) {

    FE_CHECK_ARG(values_plus.size() == values_minus.size(),
        "Plus and minus arrays must have same size");

    std::vector<Vec3> result(values_plus.size());
    for (std::size_t i = 0; i < values_plus.size(); ++i) {
        result[i] = Real(0.5) * (values_plus[i] + values_minus[i]);
    }
    return result;
}

// =============================================================================
// 2D Vector Jump and Average
// =============================================================================

std::vector<DGOperators::Vec2> DGOperators::jump_vector_2d(
    const std::vector<Vec2>& values_plus,
    const std::vector<Vec2>& values_minus) {

    FE_CHECK_ARG(values_plus.size() == values_minus.size(),
        "Plus and minus arrays must have same size");

    std::vector<Vec2> result(values_plus.size());
    for (std::size_t i = 0; i < values_plus.size(); ++i) {
        result[i] = values_plus[i] - values_minus[i];
    }
    return result;
}

std::vector<DGOperators::Vec2> DGOperators::average_vector_2d(
    const std::vector<Vec2>& values_plus,
    const std::vector<Vec2>& values_minus) {

    FE_CHECK_ARG(values_plus.size() == values_minus.size(),
        "Plus and minus arrays must have same size");

    std::vector<Vec2> result(values_plus.size());
    for (std::size_t i = 0; i < values_plus.size(); ++i) {
        result[i] = Real(0.5) * (values_plus[i] + values_minus[i]);
    }
    return result;
}

// =============================================================================
// H(div) Operators
// =============================================================================

std::vector<Real> DGOperators::normal_jump(
    const std::vector<Vec3>& values_plus,
    const std::vector<Vec3>& values_minus,
    const Vec3& normal_plus) {

    FE_CHECK_ARG(values_plus.size() == values_minus.size(),
        "Plus and minus arrays must have same size");

    std::vector<Real> result(values_plus.size());
    for (std::size_t i = 0; i < values_plus.size(); ++i) {
        // [v·n] = (v⁺ - v⁻)·n⁺ since n⁻ = -n⁺
        Vec3 diff = values_plus[i] - values_minus[i];
        result[i] = diff.dot(normal_plus);
    }
    return result;
}

std::vector<Real> DGOperators::normal_average(
    const std::vector<Vec3>& values_plus,
    const std::vector<Vec3>& values_minus,
    const Vec3& normal_plus) {

    FE_CHECK_ARG(values_plus.size() == values_minus.size(),
        "Plus and minus arrays must have same size");

    std::vector<Real> result(values_plus.size());
    for (std::size_t i = 0; i < values_plus.size(); ++i) {
        // {v·n} = ½(v⁺ + v⁻)·n⁺
        Vec3 avg = Real(0.5) * (values_plus[i] + values_minus[i]);
        result[i] = avg.dot(normal_plus);
    }
    return result;
}

// =============================================================================
// H(curl) Operators
// =============================================================================

std::vector<DGOperators::Vec3> DGOperators::tangential_jump(
    const std::vector<Vec3>& values_plus,
    const std::vector<Vec3>& values_minus,
    const Vec3& normal_plus) {

    FE_CHECK_ARG(values_plus.size() == values_minus.size(),
        "Plus and minus arrays must have same size");

    std::vector<Vec3> result(values_plus.size());
    for (std::size_t i = 0; i < values_plus.size(); ++i) {
        // [v×n] = (v⁺ - v⁻)×n⁺ since n⁻ = -n⁺
        Vec3 diff = values_plus[i] - values_minus[i];
        result[i] = math::cross(diff, normal_plus);
    }
    return result;
}

std::vector<DGOperators::Vec3> DGOperators::tangential_average(
    const std::vector<Vec3>& values_plus,
    const std::vector<Vec3>& values_minus,
    const Vec3& normal_plus) {

    FE_CHECK_ARG(values_plus.size() == values_minus.size(),
        "Plus and minus arrays must have same size");

    std::vector<Vec3> result(values_plus.size());
    for (std::size_t i = 0; i < values_plus.size(); ++i) {
        // {v×n} = ½(v⁺ + v⁻)×n⁺
        Vec3 avg = Real(0.5) * (values_plus[i] + values_minus[i]);
        result[i] = math::cross(avg, normal_plus);
    }
    return result;
}

// =============================================================================
// Gradient Operators
// =============================================================================

std::vector<Real> DGOperators::gradient_normal_jump(
    const std::vector<Vec3>& grad_plus,
    const std::vector<Vec3>& grad_minus,
    const Vec3& normal_plus) {

    FE_CHECK_ARG(grad_plus.size() == grad_minus.size(),
        "Plus and minus arrays must have same size");

    std::vector<Real> result(grad_plus.size());
    for (std::size_t i = 0; i < grad_plus.size(); ++i) {
        // [∇u·n] = (∇u⁺ - ∇u⁻)·n⁺
        Vec3 diff = grad_plus[i] - grad_minus[i];
        result[i] = diff.dot(normal_plus);
    }
    return result;
}

std::vector<Real> DGOperators::gradient_normal_average(
    const std::vector<Vec3>& grad_plus,
    const std::vector<Vec3>& grad_minus,
    const Vec3& normal_plus) {

    FE_CHECK_ARG(grad_plus.size() == grad_minus.size(),
        "Plus and minus arrays must have same size");

    std::vector<Real> result(grad_plus.size());
    for (std::size_t i = 0; i < grad_plus.size(); ++i) {
        // {∇u·n} = ½(∇u⁺ + ∇u⁻)·n⁺
        Vec3 avg = Real(0.5) * (grad_plus[i] + grad_minus[i]);
        result[i] = avg.dot(normal_plus);
    }
    return result;
}

// =============================================================================
// Penalty Utilities
// =============================================================================

Real DGOperators::penalty_parameter(
    int polynomial_order,
    Real mesh_size,
    Real constant) {

    FE_CHECK_ARG(polynomial_order >= 0, "Polynomial order must be non-negative");
    FE_CHECK_ARG(mesh_size > Real(0), "Mesh size must be positive");

    const Real p = static_cast<Real>(polynomial_order);
    // η = C · (p+1)² / h  (using p+1 to handle p=0 case)
    return constant * (p + Real(1)) * (p + Real(1)) / mesh_size;
}

Real DGOperators::harmonic_average(Real kappa_plus, Real kappa_minus) {
    const Real sum = kappa_plus + kappa_minus;
    if (std::abs(sum) < Real(1e-14)) {
        return Real(0);
    }
    return Real(2) * kappa_plus * kappa_minus / sum;
}

// =============================================================================
// Upwind Operators
// =============================================================================

std::vector<Real> DGOperators::upwind(
    const std::vector<Real>& values_plus,
    const std::vector<Real>& values_minus,
    const Vec3& velocity,
    const Vec3& normal_plus) {

    FE_CHECK_ARG(values_plus.size() == values_minus.size(),
        "Plus and minus arrays must have same size");

    const Real vn = velocity.dot(normal_plus);
    const Real tol = Real(1e-14);

    std::vector<Real> result(values_plus.size());

    if (vn > tol) {
        // Flow leaves plus element - upwind from plus
        for (std::size_t i = 0; i < values_plus.size(); ++i) {
            result[i] = values_plus[i];
        }
    } else if (vn < -tol) {
        // Flow enters plus element - upwind from minus
        for (std::size_t i = 0; i < values_plus.size(); ++i) {
            result[i] = values_minus[i];
        }
    } else {
        // Flow parallel to face - use average
        for (std::size_t i = 0; i < values_plus.size(); ++i) {
            result[i] = Real(0.5) * (values_plus[i] + values_minus[i]);
        }
    }

    return result;
}

std::vector<Real> DGOperators::lax_friedrichs_flux(
    const std::vector<Real>& values_plus,
    const std::vector<Real>& values_minus,
    const std::vector<Real>& flux_plus,
    const std::vector<Real>& flux_minus,
    Real max_wave_speed) {

    FE_CHECK_ARG(values_plus.size() == values_minus.size(),
        "Value arrays must have same size");
    FE_CHECK_ARG(flux_plus.size() == flux_minus.size(),
        "Flux arrays must have same size");
    FE_CHECK_ARG(values_plus.size() == flux_plus.size(),
        "Value and flux arrays must have same size");

    std::vector<Real> result(values_plus.size());
    const Real half_lambda = Real(0.5) * max_wave_speed;

    for (std::size_t i = 0; i < values_plus.size(); ++i) {
        // F_LF = {F} - ½·λ·[u] = ½(F⁺ + F⁻) - ½·λ·(u⁺ - u⁻)
        const Real flux_avg = Real(0.5) * (flux_plus[i] + flux_minus[i]);
        const Real jump_val = values_plus[i] - values_minus[i];
        result[i] = flux_avg - half_lambda * jump_val;
    }

    return result;
}

} // namespace spaces
} // namespace FE
} // namespace svmp
