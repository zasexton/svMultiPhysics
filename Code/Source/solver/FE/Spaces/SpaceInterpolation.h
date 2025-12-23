/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_SPACEINTERPOLATION_H
#define SVMP_FE_SPACES_SPACEINTERPOLATION_H

/**
 * @file SpaceInterpolation.h
 * @brief Interpolation and projection between function spaces
 */

#include "Spaces/FunctionSpace.h"
#include "Basis/LagrangeBasis.h"

namespace svmp {
namespace FE {
namespace spaces {

/**
 * @brief Utilities for transferring fields between function spaces
 *
 * All operations are element-local and reference-space only; no Mesh
 * connectivity is required. L² projection works for scalar and vector
 * spaces. Nodal interpolation is implemented for scalar Lagrange bases.
 */
class SpaceInterpolation {
public:
    /**
     * @brief L² projection of a field between spaces on the same element
     *
     * Projects the field represented by (src_space, src_coeffs) into the
     * target space dst_space using the mass-matrix based interpolation
     * implemented by FunctionSpace::interpolate.
     */
    static void l2_projection(const FunctionSpace& src_space,
                              const std::vector<Real>& src_coeffs,
                              const FunctionSpace& dst_space,
                              std::vector<Real>& dst_coeffs);

    /**
     * @brief Nodal interpolation for scalar Lagrange spaces
     *
     * Interpolates a scalar field represented in src_space to dst_space by
     * evaluating the source field at the Lagrange nodes of the destination
     * basis. Requires both spaces to use LagrangeBasis and scalar fields.
     */
    static void nodal_interpolation(const FunctionSpace& src_space,
                                    const std::vector<Real>& src_coeffs,
                                    const FunctionSpace& dst_space,
                                    std::vector<Real>& dst_coeffs);

    /// Alias for conservative interpolation: currently implemented as L² projection
    static void conservative_interpolation(const FunctionSpace& src_space,
                                           const std::vector<Real>& src_coeffs,
                                           const FunctionSpace& dst_space,
                                           std::vector<Real>& dst_coeffs) {
        l2_projection(src_space, src_coeffs, dst_space, dst_coeffs);
    }
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_SPACEINTERPOLATION_H

