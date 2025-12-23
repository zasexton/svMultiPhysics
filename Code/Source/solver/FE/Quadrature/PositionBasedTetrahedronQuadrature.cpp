/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "PositionBasedTetrahedronQuadrature.h"
#include "Core/FEException.h"
#include <cmath>

namespace svmp {
namespace FE {
namespace quadrature {

PositionBasedTetrahedronQuadrature::PositionBasedTetrahedronQuadrature(
    const PositionBasedParams& params)
    : QuadratureRule(svmp::CellFamily::Tetra, 3, compute_order(params.modifier))
    , modifier_(params.modifier)
{
    initialize(params.modifier);
}

PositionBasedTetrahedronQuadrature::PositionBasedTetrahedronQuadrature(Real modifier)
    : QuadratureRule(svmp::CellFamily::Tetra, 3, compute_order(modifier))
    , modifier_(modifier)
{
    initialize(modifier);
}

void PositionBasedTetrahedronQuadrature::initialize(Real s) {
    // Validate range: s must be in [0.25, 1.0]
    const Real min_s = 0.25 - 1e-14;  // Small tolerance for floating point
    const Real max_s = 1.0 + 1e-14;

    if (s < min_s || s > max_s) {
        throw FEException(
            "PositionBasedTetrahedronQuadrature: modifier must be in [0.25, 1.0], got " +
            std::to_string(s),
            __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // Compute the complementary coordinate
    // t = (1-s)/3
    // When s = 0.25: t = 0.25 (centroid, all coordinates equal)
    // When s ≈ 0.585: t ≈ 0.138 (Gaussian)
    // When s = 1.0: t = 0 (vertices)
    Real t = (1.0 - s) / 3.0;

    // Create the 4 quadrature points
    // These are positioned symmetrically in the reference tetrahedron
    // Each point is "pulled" toward one of the four vertices
    std::vector<QuadPoint> pts(4);
    pts[0] = {s, t, t};    // Point near vertex (1,0,0) or centroid
    pts[1] = {t, s, t};    // Point near vertex (0,1,0) or centroid
    pts[2] = {t, t, s};    // Point near vertex (0,0,1) or centroid
    pts[3] = {t, t, t};    // Point near vertex (0,0,0) or centroid

    // Weights are always 1/24 each, summing to 1/6 (the reference tet volume)
    // This is independent of the point positions (legacy behavior)
    std::vector<Real> wts(4, 1.0 / 24.0);

    set_data(std::move(pts), std::move(wts));
}

int PositionBasedTetrahedronQuadrature::compute_order(Real s) noexcept {
    // The standard Gaussian rule achieves polynomial exactness of order 2
    // meaning it can exactly integrate polynomials of total degree <= 2.
    //
    // The Gaussian position is s = (5 + 3*sqrt(5))/20 ≈ 0.5854101966249685
    //
    // For non-Gaussian positions, the exactness is reduced:
    // - Central rule (s = 0.25) only integrates constants exactly
    // - Nodal rule (s = 1.0) only integrates constants exactly
    //
    // We use a heuristic based on distance from the Gaussian position.

    const Real gaussian_s = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
    const Real tolerance = 0.01;

    if (std::abs(s - gaussian_s) < tolerance) {
        return 2;  // Full polynomial exactness for Gaussian position
    }

    // For positions significantly different from Gaussian, reduce reported order
    return 1;
}

bool PositionBasedTetrahedronQuadrature::is_gaussian() const noexcept {
    const Real gaussian_s = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
    return std::abs(modifier_ - gaussian_s) < 1e-10;
}

bool PositionBasedTetrahedronQuadrature::is_central() const noexcept {
    return std::abs(modifier_ - 0.25) < 1e-10;
}

bool PositionBasedTetrahedronQuadrature::is_nodal() const noexcept {
    return std::abs(modifier_ - 1.0) < 1e-10;
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
