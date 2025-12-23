/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_DGOPERATORS_H
#define SVMP_FE_SPACES_DGOPERATORS_H

/**
 * @file DGOperators.h
 * @brief Discontinuous Galerkin operators for interior face computations
 *
 * This module provides static utilities for DG methods that require
 * computing jumps and averages across interior element faces. These
 * operators are fundamental to:
 *  - Interior penalty DG (IPDG/SIPG/NIPG)
 *  - Local DG (LDG)
 *  - Hybridizable DG (HDG)
 *  - Flux reconstruction methods
 *
 * Notation:
 *  - [v] = v⁺ - v⁻ : Jump operator
 *  - {v} = ½(v⁺ + v⁻) : Average operator
 *  - v⁺ : Value from element on "plus" side of face
 *  - v⁻ : Value from element on "minus" side of face
 *
 * The choice of plus/minus side is arbitrary but must be consistent.
 * Typically the plus side is the element with the larger global ID.
 */

#include "Core/Types.h"
#include "Math/Vector.h"
#include <vector>
#include <array>

namespace svmp {
namespace FE {
namespace spaces {

/**
 * @brief Static utility class for DG face operators
 *
 * All methods are stateless and operate on arrays of values at
 * face quadrature points. The plus/minus convention must be
 * consistent across an assembly loop.
 */
class DGOperators {
public:
    using Vec3 = math::Vector<Real, 3>;
    using Vec2 = math::Vector<Real, 2>;

    // =========================================================================
    // Scalar Jump and Average Operators
    // =========================================================================

    /**
     * @brief Scalar jump: [v] = v⁺ - v⁻
     *
     * @param values_plus Values from plus element at face quadrature points
     * @param values_minus Values from minus element at face quadrature points
     * @return Jump values at quadrature points
     */
    static std::vector<Real> jump(
        const std::vector<Real>& values_plus,
        const std::vector<Real>& values_minus);

    /**
     * @brief Scalar average: {v} = ½(v⁺ + v⁻)
     *
     * @param values_plus Values from plus element
     * @param values_minus Values from minus element
     * @return Average values at quadrature points
     */
    static std::vector<Real> average(
        const std::vector<Real>& values_plus,
        const std::vector<Real>& values_minus);

    /**
     * @brief Weighted average: {v}_w = w⁺·v⁺ + w⁻·v⁻
     *
     * Useful for harmonic averaging of diffusion coefficients:
     *   κ_avg = 2·κ⁺·κ⁻/(κ⁺ + κ⁻)
     *
     * @param values_plus Values from plus element
     * @param values_minus Values from minus element
     * @param weight_plus Weight for plus side (w⁺)
     * @param weight_minus Weight for minus side (w⁻ = 1 - w⁺ for averaging)
     * @return Weighted average values
     */
    static std::vector<Real> weighted_average(
        const std::vector<Real>& values_plus,
        const std::vector<Real>& values_minus,
        Real weight_plus,
        Real weight_minus);

    // =========================================================================
    // Vector Jump and Average Operators (3D)
    // =========================================================================

    /**
     * @brief Vector jump: [v] = v⁺ - v⁻ (component-wise)
     *
     * @param values_plus Vector values from plus element [npts]
     * @param values_minus Vector values from minus element [npts]
     * @return Jump vectors at quadrature points [npts]
     */
    static std::vector<Vec3> jump_vector(
        const std::vector<Vec3>& values_plus,
        const std::vector<Vec3>& values_minus);

    /**
     * @brief Vector average: {v} = ½(v⁺ + v⁻) (component-wise)
     *
     * @param values_plus Vector values from plus element
     * @param values_minus Vector values from minus element
     * @return Average vectors at quadrature points
     */
    static std::vector<Vec3> average_vector(
        const std::vector<Vec3>& values_plus,
        const std::vector<Vec3>& values_minus);

    // =========================================================================
    // Vector Jump and Average Operators (2D)
    // =========================================================================

    /**
     * @brief 2D Vector jump: [v] = v⁺ - v⁻
     */
    static std::vector<Vec2> jump_vector_2d(
        const std::vector<Vec2>& values_plus,
        const std::vector<Vec2>& values_minus);

    /**
     * @brief 2D Vector average: {v} = ½(v⁺ + v⁻)
     */
    static std::vector<Vec2> average_vector_2d(
        const std::vector<Vec2>& values_plus,
        const std::vector<Vec2>& values_minus);

    // =========================================================================
    // H(div) Specific Operators
    // =========================================================================

    /**
     * @brief Normal jump for H(div): [v·n] = v⁺·n⁺ + v⁻·n⁻
     *
     * For H(div) conforming spaces, the normal component is continuous,
     * so this should be zero. This operator is useful for:
     *  - Verifying H(div) conformity
     *  - Non-conforming discretizations
     *  - Penalization terms
     *
     * Note: n⁺ = -n⁻ by convention (outward normals), so
     *       [v·n] = v⁺·n⁺ - v⁻·n⁺ = (v⁺ - v⁻)·n⁺
     *
     * @param values_plus Vector values from plus element
     * @param values_minus Vector values from minus element
     * @param normal_plus Outward normal from plus element (n⁺)
     * @return Normal jump values
     */
    static std::vector<Real> normal_jump(
        const std::vector<Vec3>& values_plus,
        const std::vector<Vec3>& values_minus,
        const Vec3& normal_plus);

    /**
     * @brief Normal average for H(div): {v·n} = ½(v⁺·n⁺ - v⁻·n⁺)
     *
     * Since n⁻ = -n⁺, this becomes:
     *   {v·n} = ½(v⁺ + v⁻)·n⁺
     *
     * @param values_plus Vector values from plus element
     * @param values_minus Vector values from minus element
     * @param normal_plus Outward normal from plus element
     * @return Normal average values
     */
    static std::vector<Real> normal_average(
        const std::vector<Vec3>& values_plus,
        const std::vector<Vec3>& values_minus,
        const Vec3& normal_plus);

    // =========================================================================
    // H(curl) Specific Operators
    // =========================================================================

    /**
     * @brief Tangential jump for H(curl): [v×n] = v⁺×n⁺ + v⁻×n⁻
     *
     * For H(curl) conforming spaces, the tangential trace is continuous,
     * so [v×n] should be zero. This operator is useful for:
     *  - Verifying H(curl) conformity
     *  - DG discretizations of Maxwell's equations
     *  - Penalization terms
     *
     * Since n⁻ = -n⁺:
     *   [v×n] = v⁺×n⁺ - v⁻×n⁺ = (v⁺ - v⁻)×n⁺
     *
     * @param values_plus Vector values from plus element
     * @param values_minus Vector values from minus element
     * @param normal_plus Outward normal from plus element
     * @return Tangential jump vectors
     */
    static std::vector<Vec3> tangential_jump(
        const std::vector<Vec3>& values_plus,
        const std::vector<Vec3>& values_minus,
        const Vec3& normal_plus);

    /**
     * @brief Tangential average for H(curl): {v×n} = ½(v⁺×n⁺ - v⁻×n⁺)
     *
     * Since n⁻ = -n⁺:
     *   {v×n} = ½(v⁺ + v⁻)×n⁺
     *
     * @param values_plus Vector values from plus element
     * @param values_minus Vector values from minus element
     * @param normal_plus Outward normal from plus element
     * @return Tangential average vectors
     */
    static std::vector<Vec3> tangential_average(
        const std::vector<Vec3>& values_plus,
        const std::vector<Vec3>& values_minus,
        const Vec3& normal_plus);

    // =========================================================================
    // Gradient Jump/Average (for diffusion problems)
    // =========================================================================

    /**
     * @brief Gradient jump dotted with normal: [∇u·n]
     *
     * Computes the jump of the normal derivative across the face:
     *   [∇u·n] = ∇u⁺·n⁺ + ∇u⁻·n⁻ = (∇u⁺ - ∇u⁻)·n⁺
     *
     * @param grad_plus Gradients from plus element [npts]
     * @param grad_minus Gradients from minus element [npts]
     * @param normal_plus Outward normal from plus element
     * @return Normal gradient jumps
     */
    static std::vector<Real> gradient_normal_jump(
        const std::vector<Vec3>& grad_plus,
        const std::vector<Vec3>& grad_minus,
        const Vec3& normal_plus);

    /**
     * @brief Gradient average dotted with normal: {∇u·n}
     *
     * @param grad_plus Gradients from plus element
     * @param grad_minus Gradients from minus element
     * @param normal_plus Outward normal from plus element
     * @return Normal gradient averages
     */
    static std::vector<Real> gradient_normal_average(
        const std::vector<Vec3>& grad_plus,
        const std::vector<Vec3>& grad_minus,
        const Vec3& normal_plus);

    // =========================================================================
    // Penalty Parameter Utilities
    // =========================================================================

    /**
     * @brief Compute interior penalty parameter
     *
     * For SIPG/IPDG methods, the penalty parameter is typically:
     *   η = C · p² / h
     * where C is a constant, p is polynomial order, h is mesh size.
     *
     * @param polynomial_order Polynomial order p
     * @param mesh_size Local mesh size h (e.g., face diameter)
     * @param constant Penalty constant C (typically 1-10)
     * @return Penalty parameter η
     */
    static Real penalty_parameter(
        int polynomial_order,
        Real mesh_size,
        Real constant = Real(1));

    /**
     * @brief Compute harmonic average of diffusion coefficients
     *
     * For interfaces with different material properties:
     *   κ_avg = 2·κ⁺·κ⁻ / (κ⁺ + κ⁻)
     *
     * @param kappa_plus Diffusivity on plus side
     * @param kappa_minus Diffusivity on minus side
     * @return Harmonic average
     */
    static Real harmonic_average(Real kappa_plus, Real kappa_minus);

    // =========================================================================
    // Upwind Operators (for advection)
    // =========================================================================

    /**
     * @brief Upwind value selection based on advection velocity
     *
     * Returns the upwind value based on the sign of velocity·normal:
     *   - If v·n > 0: upwind from plus side (flow leaves plus element)
     *   - If v·n < 0: upwind from minus side (flow enters plus element)
     *   - If v·n ≈ 0: average
     *
     * @param values_plus Values from plus element
     * @param values_minus Values from minus element
     * @param velocity Advection velocity
     * @param normal_plus Normal from plus element
     * @return Upwind values
     */
    static std::vector<Real> upwind(
        const std::vector<Real>& values_plus,
        const std::vector<Real>& values_minus,
        const Vec3& velocity,
        const Vec3& normal_plus);

    /**
     * @brief Lax-Friedrichs numerical flux
     *
     * Computes the Lax-Friedrichs flux:
     *   F_LF = {F(u)} - ½·λ·[u]
     * where λ is the maximum wave speed.
     *
     * @param values_plus Values from plus element
     * @param values_minus Values from minus element
     * @param flux_plus Flux evaluated at plus values
     * @param flux_minus Flux evaluated at minus values
     * @param max_wave_speed Maximum wave speed λ
     * @return Lax-Friedrichs flux
     */
    static std::vector<Real> lax_friedrichs_flux(
        const std::vector<Real>& values_plus,
        const std::vector<Real>& values_minus,
        const std::vector<Real>& flux_plus,
        const std::vector<Real>& flux_minus,
        Real max_wave_speed);
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_DGOPERATORS_H
