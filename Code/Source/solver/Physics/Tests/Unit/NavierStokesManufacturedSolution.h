/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_TESTS_UNIT_NAVIER_STOKES_MANUFACTURED_SOLUTION_H
#define SVMP_PHYSICS_TESTS_UNIT_NAVIER_STOKES_MANUFACTURED_SOLUTION_H

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "FE/Quadrature/TriangleQuadrature.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace svmp::Physics::test::ns_mms {

using Vector = std::array<double, 2>;
using Gradient = std::array<Vector, 2>;  // [velocity component][spatial derivative]

inline constexpr double kDensity = 1.0;
inline constexpr double kViscosity = 0.01;
inline constexpr double kVelocityFrequency = 2.0 * std::numbers::pi;
inline constexpr double kPressureFrequency = 4.0;

[[nodiscard]] inline double velocityAmplitude(double t)
{
    const double s = std::sin(0.5 * kVelocityFrequency * t);
    return 2.0 * s * s;  // 1 - cos(2 pi t), including near startup
}

[[nodiscard]] inline double velocityAmplitudeRate(double t)
{
    return kVelocityFrequency * std::sin(kVelocityFrequency * t);
}

[[nodiscard]] inline double pressureAmplitude(double t)
{
    return std::sin(kPressureFrequency * t);
}

namespace detail {

// B(s) = s^2 (1-s)^2 and its first three derivatives.
template<class T>
struct Bubble {
    T value, first, second, third;
};

template<class T>
[[nodiscard]] Bubble<T> bubble(const T& s)
{
    const auto product = s * (1.0 - s);
    return {product * product, 2.0 * product * (1.0 - 2.0 * s),
            2.0 - 12.0 * product, 24.0 * s - 12.0};
}

template<class T>
struct VelocityMode {
    std::array<T, 2> value;
    std::array<std::array<T, 2>, 2> gradient;
    std::array<T, 2> laplacian;

    [[nodiscard]] std::array<T, 2> velocity(const T& amplitude) const
    {
        return {1.0 + amplitude * value[0], amplitude * value[1]};
    }
};

// The curl of 32 B(x) B(y) is solenoidal and vanishes on the square's boundary.
template<class T>
[[nodiscard]] VelocityMode<T> velocityMode(const T& x, const T& y)
{
    const auto bx = bubble(x);
    const auto by = bubble(y);
    const auto dx_vx = 32.0 * bx.first * by.first;
    return {
        {32.0 * bx.value * by.first, -32.0 * bx.first * by.value},
        {{{dx_vx, 32.0 * bx.value * by.second},
          {-32.0 * bx.second * by.value, -dx_vx}}},
        {32.0 * (bx.second * by.first + bx.value * by.third),
         -32.0 * (bx.third * by.value + bx.first * by.second)}
    };
}

template<class T>
[[nodiscard]] std::array<T, 2> pressureShapeGradient(const T& x, const T& y)
{
    return {2.0 * x + y, x - 1.0};
}

} // namespace detail

[[nodiscard]] inline Vector velocity(double x, double y, double t)
{
    return detail::velocityMode(x, y).velocity(velocityAmplitude(t));
}

[[nodiscard]] inline Vector velocityRate(double x, double y, double t)
{
    const auto v = detail::velocityMode(x, y).value;
    const double rate = velocityAmplitudeRate(t);
    return {rate * v[0], rate * v[1]};
}

[[nodiscard]] inline double pressureShape(double x, double y)
{
    return x * x + (x - 1.0) * y;
}

[[nodiscard]] inline double pressure(double x, double y, double t)
{
    return pressureAmplitude(t) * pressureShape(x, y);
}

// The same algebra accepts scalar coordinates and symbolic form expressions.
// f = du/dt + (u . grad)u - (mu/rho) Laplacian(u) + grad(p)/rho.
template<class T>
[[nodiscard]] std::array<T, 2> bodyAcceleration(const T& x, const T& y,
                                               const T& amplitude, const T& amplitude_rate,
                                               const T& pressure_amplitude)
{
    const auto mode = detail::velocityMode(x, y);
    const auto u = mode.velocity(amplitude);
    const auto grad_p = detail::pressureShapeGradient(x, y);
    std::array<T, 2> force;
    for (int d = 0; d < 2; ++d) {
        const auto convection = u[0] * mode.gradient[d][0] + u[1] * mode.gradient[d][1];
        force[d] = amplitude_rate * mode.value[d] +
                   amplitude * (convection - (kViscosity / kDensity) * mode.laplacian[d]) +
                   (pressure_amplitude / kDensity) * grad_p[d];
    }
    return force;
}

struct ErrorNorms {
    double velocity_l2{};
    double velocity_h1{};  // full H1 norm, including velocity_l2
    double pressure_l2{};
    double divergence_l2{};
    std::vector<double> nodal_pressure_error;  // mean-aligned p_h minus p_exact
};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

// P1 nodal fields on the serial unit square. The pressure unknown belongs to
// the generalized-alpha stage, while velocity has been recovered at the endpoint.
[[nodiscard]] inline ErrorNorms measureErrors(const Mesh& mesh,
                                              std::span<const Vector> endpoint_velocity,
                                              std::span<const double> stage_pressure,
                                              double velocity_time, double pressure_time)
{
    const auto& base = mesh.base();
    if (base.dim() != 2 || base.n_cells() == 0 || base.n_vertices() == 0 ||
        endpoint_velocity.size() != base.n_vertices() || stage_pressure.size() != base.n_vertices()) {
        throw std::invalid_argument("Expected a 2D triangle mesh with velocity and pressure at every vertex");
    }

    // Integrate the P1 mean exactly. Subtract a common nodal value first to
    // preserve pressure differences when the numerical gauge is large.
    const double reference_pressure = stage_pressure.front();
    long double centered_integral = 0.0L;
    long double total_area = 0.0L;
    for (index_t cell = 0; cell < static_cast<index_t>(base.n_cells()); ++cell) {
        const auto [vertices, count] = base.cell_vertices_span(cell);
        if (count != 3 || base.cell_shape(cell).family != CellFamily::Triangle) {
            throw std::invalid_argument("Pressure and velocity errors require Tri3 cells");
        }
        const double area = base.cell_measure(cell);
        if (!(area > 0.0)) {
            throw std::invalid_argument("Degenerate triangle in manufactured-solution mesh");
        }
        double centered_sum = 0.0;
        for (int i = 0; i < 3; ++i) {
            centered_sum += stage_pressure[vertices[i]] - reference_pressure;
        }
        centered_integral += static_cast<long double>(area) * centered_sum / 3.0L;
        total_area += area;
    }
    const double pressure_amplitude = pressureAmplitude(pressure_time);
    // Integral of x^2 + xy - y over the unit square is 1/3 + 1/4 - 1/2.
    const double centered_offset = static_cast<double>(centered_integral / total_area) -
                                   pressure_amplitude / 12.0;
    ErrorNorms errors;
    errors.nodal_pressure_error.resize(base.n_vertices());
    for (index_t v = 0; v < static_cast<index_t>(base.n_vertices()); ++v) {
        const auto point = base.get_vertex_coords(v);
        errors.nodal_pressure_error[v] = (stage_pressure[v] - reference_pressure) - centered_offset -
                                        pressure_amplitude * pressureShape(point[0], point[1]);
    }

    // Each exact velocity component has total degree seven. This rule
    // integrates the squared P1 velocity error and all lower-degree terms.
    static const FE::quadrature::TriangleQuadrature rule(14);
    const double amplitude = velocityAmplitude(velocity_time);
    double velocity_squared = 0.0;
    double gradient_squared = 0.0;
    double pressure_squared = 0.0;
    double divergence_squared = 0.0;
    for (index_t cell = 0; cell < static_cast<index_t>(base.n_cells()); ++cell) {
        const auto* v = base.cell_vertices_span(cell).first;
        const auto x0 = base.get_vertex_coords(v[0]);
        const auto x1 = base.get_vertex_coords(v[1]);
        const auto x2 = base.get_vertex_coords(v[2]);
        const Vector e1{x1[0] - x0[0], x1[1] - x0[1]};
        const Vector e2{x2[0] - x0[0], x2[1] - x0[1]};
        const double det = e1[0] * e2[1] - e1[1] * e2[0];
        const double abs_det = std::abs(det);
        const auto& u0 = endpoint_velocity[v[0]];
        Vector du1, du2;
        Gradient grad_uh;
        for (int d = 0; d < 2; ++d) {
            du1[d] = endpoint_velocity[v[1]][d] - u0[d];
            du2[d] = endpoint_velocity[v[2]][d] - u0[d];
            grad_uh[d] = {(du1[d] * e2[1] - du2[d] * e1[1]) / det,
                          (du2[d] * e1[0] - du1[d] * e2[0]) / det};
        }
        const double div_uh = grad_uh[0][0] + grad_uh[1][1];
        divergence_squared += 0.5 * abs_det * div_uh * div_uh;

        const double p0 = stage_pressure[v[0]];
        const double aligned_p0 = (p0 - reference_pressure) - centered_offset;
        const double dp1 = stage_pressure[v[1]] - p0;
        const double dp2 = stage_pressure[v[2]] - p0;
        for (std::size_t q = 0; q < rule.num_points(); ++q) {
            const auto& rs = rule.point(q);
            const double weight = rule.weight(q) * abs_det;
            const double x = x0[0] + rs[0] * e1[0] + rs[1] * e2[0];
            const double y = x0[1] + rs[0] * e1[1] + rs[1] * e2[1];
            const auto mode = detail::velocityMode(x, y);
            const auto exact_u = mode.velocity(amplitude);
            for (int d = 0; d < 2; ++d) {
                const double uh = u0[d] + rs[0] * du1[d] + rs[1] * du2[d];
                const double delta = uh - exact_u[d];
                velocity_squared += weight * delta * delta;
                for (int axis = 0; axis < 2; ++axis) {
                    const double dg = grad_uh[d][axis] - amplitude * mode.gradient[d][axis];
                    gradient_squared += weight * dg * dg;
                }
            }
            const double ph = aligned_p0 + rs[0] * dp1 + rs[1] * dp2;
            const double delta_p = ph - pressure_amplitude * pressureShape(x, y);
            pressure_squared += weight * delta_p * delta_p;
        }
    }
    errors.velocity_l2 = std::sqrt(velocity_squared);
    errors.velocity_h1 = std::sqrt(velocity_squared + gradient_squared);
    errors.pressure_l2 = std::sqrt(pressure_squared);
    errors.divergence_l2 = std::sqrt(divergence_squared);
    return errors;
}

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

struct SpatialPressureCheck {
    double ripple{};
    int extrema{};
    double largest_reversal{};
};

// On each interior row the exact pressure increases with x, and on each
// interior column it decreases with y (0 < t <= 0.25). An axial extremum
// of p_h therefore signals a reversal absent from the manufactured field.
[[nodiscard]] inline SpatialPressureCheck checkPressure(std::span<const double> nodal_pressure_error,
                                                       std::span<const double> stage_pressure, int subdivisions)
{
    const auto width = static_cast<std::size_t>(subdivisions) + 1;
    if (subdivisions < 2 || nodal_pressure_error.size() != width * width ||
        stage_pressure.size() != width * width) {
        throw std::invalid_argument("Pressure check requires a square grid with interior vertices");
    }
    SpatialPressureCheck check;
    double squared = 0.0;
    for (std::size_t j = 1; j + 1 < width; ++j) {
        for (std::size_t i = 1; i + 1 < width; ++i) {
            const auto k = j * width + i;
            const double center = nodal_pressure_error[k];
            const double ripple = 0.25 * ((center - nodal_pressure_error[k - 1]) +
                                          (center - nodal_pressure_error[k + 1]) +
                                          (center - nodal_pressure_error[k - width]) +
                                          (center - nodal_pressure_error[k + width]));
            squared += ripple * ripple;
            for (const auto stride : {std::size_t{1}, width}) {
                const double before = stage_pressure[k] - stage_pressure[k - stride];
                const double after = stage_pressure[k + stride] - stage_pressure[k];
                if (before * after < 0.0) {
                    ++check.extrema;
                    check.largest_reversal = std::max(check.largest_reversal,
                                                      std::min(std::abs(before), std::abs(after)));
                }
            }
        }
    }
    const auto interior_width = width - 2;
    check.ripple = std::sqrt(squared / static_cast<double>(interior_width * interior_width));
    return check;
}

} // namespace svmp::Physics::test::ns_mms

#endif // SVMP_PHYSICS_TESTS_UNIT_NAVIER_STOKES_MANUFACTURED_SOLUTION_H
