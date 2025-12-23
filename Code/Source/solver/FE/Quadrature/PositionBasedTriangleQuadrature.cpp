/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "PositionBasedTriangleQuadrature.h"
#include "Core/FEException.h"
#include <cmath>

namespace svmp {
namespace FE {
namespace quadrature {

PositionBasedTriangleQuadrature::PositionBasedTriangleQuadrature(
    const PositionBasedParams& params)
    : QuadratureRule(svmp::CellFamily::Triangle, 2, compute_order(params.modifier))
    , modifier_(params.modifier)
{
    initialize(params.modifier);
}

PositionBasedTriangleQuadrature::PositionBasedTriangleQuadrature(Real modifier)
    : QuadratureRule(svmp::CellFamily::Triangle, 2, compute_order(modifier))
    , modifier_(modifier)
{
    initialize(modifier);
}

void PositionBasedTriangleQuadrature::initialize(Real s) {
    // Validate range: s must be in [1/3, 1.0]
    const Real min_s = 1.0 / 3.0 - 1e-14;  // Small tolerance for floating point
    const Real max_s = 1.0 + 1e-14;

    if (s < min_s || s > max_s) {
        throw FEException(
            "PositionBasedTriangleQuadrature: modifier must be in [1/3, 1.0], got " +
            std::to_string(s),
            __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // Compute the complementary coordinate
    // t = -0.5*s + 0.5
    // When s = 1/3: t = 1/3 (centroid)
    // When s = 2/3: t = 1/6 (Gaussian)
    // When s = 1.0: t = 0   (vertices)
    Real t = -0.5 * s + 0.5;

    // Create the 3 quadrature points
    // These are positioned symmetrically in the reference triangle
    std::vector<QuadPoint> pts(3);
    pts[0] = {t, t, 0.0};    // Point near vertex (0,0) or centroid
    pts[1] = {s, t, 0.0};    // Point near vertex (1,0) or centroid
    pts[2] = {t, s, 0.0};    // Point near vertex (0,1) or centroid

    // Weights are always 1/6 each, summing to 0.5 (the reference triangle area)
    // This is independent of the point positions (legacy behavior)
    std::vector<Real> wts(3, 1.0 / 6.0);

    set_data(std::move(pts), std::move(wts));
}

int PositionBasedTriangleQuadrature::compute_order(Real s) noexcept {
    // The standard Gaussian rule (s = 2/3) achieves polynomial exactness of order 2
    // meaning it can exactly integrate polynomials of total degree <= 2.
    //
    // For non-Gaussian positions, the exactness is reduced:
    // - Central rule (s = 1/3) only integrates constants exactly
    // - Nodal rule (s = 1.0) only integrates constants exactly
    //
    // We use a heuristic based on distance from the Gaussian position.

    const Real gaussian_s = 2.0 / 3.0;
    const Real tolerance = 0.01;

    if (std::abs(s - gaussian_s) < tolerance) {
        return 2;  // Full polynomial exactness for Gaussian position
    }

    // For positions significantly different from Gaussian, reduce reported order
    return 1;
}

bool PositionBasedTriangleQuadrature::is_gaussian() const noexcept {
    const Real gaussian_s = 2.0 / 3.0;
    return std::abs(modifier_ - gaussian_s) < 1e-10;
}

bool PositionBasedTriangleQuadrature::is_central() const noexcept {
    const Real central_s = 1.0 / 3.0;
    return std::abs(modifier_ - central_s) < 1e-10;
}

bool PositionBasedTriangleQuadrature::is_nodal() const noexcept {
    return std::abs(modifier_ - 1.0) < 1e-10;
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
