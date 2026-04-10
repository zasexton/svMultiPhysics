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

#include <functional>
#include <span>

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
    struct TransferOperator {
        std::size_t rows{0};
        std::size_t cols{0};
        std::vector<Real> values;

        [[nodiscard]] Real operator()(std::size_t row,
                                      std::size_t col) const {
            FE_CHECK_ARG(row < rows && col < cols,
                         "SpaceInterpolation::TransferOperator index out of range");
            return values[row * cols + col];
        }

        [[nodiscard]] Real& operator()(std::size_t row,
                                       std::size_t col) {
            FE_CHECK_ARG(row < rows && col < cols,
                         "SpaceInterpolation::TransferOperator index out of range");
            return values[row * cols + col];
        }
    };

    using ReferencePointMap = std::function<FunctionSpace::Value(const FunctionSpace::Value&)>;
    using ValueTransform = std::function<FunctionSpace::Value(const FunctionSpace::Value&)>;

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

    /**
     * @brief Build a dense element-local transfer operator from src_space to dst_space
     *
     * The operator maps source coefficients to destination coefficients using
     * the same interpolation machinery as @ref l2_projection. An optional
     * reference-point map can be provided to evaluate the source space in a
     * different face/edge orientation than the destination space.
     */
    static TransferOperator build_transfer_operator(
        const FunctionSpace& src_space,
        const FunctionSpace& dst_space,
        const ReferencePointMap& point_map = {},
        const ValueTransform& value_transform = {});

    /// Dense prolongation operator from coarse to fine
    static TransferOperator prolongation_operator(
        const FunctionSpace& coarse_space,
        const FunctionSpace& fine_space,
        const ReferencePointMap& point_map = {},
        const ValueTransform& value_transform = {}) {
        return build_transfer_operator(coarse_space, fine_space, point_map, value_transform);
    }

    /// Dense restriction operator from fine to coarse
    static TransferOperator restriction_operator(
        const FunctionSpace& fine_space,
        const FunctionSpace& coarse_space,
        const ReferencePointMap& point_map = {},
        const ValueTransform& value_transform = {}) {
        return build_transfer_operator(fine_space, coarse_space, point_map, value_transform);
    }

    /// Apply a dense transfer operator to a coefficient vector
    static void apply_transfer(const TransferOperator& op,
                               std::span<const Real> src_coeffs,
                               std::vector<Real>& dst_coeffs);

    /// Apply prolongation from coarse to fine coefficients
    static void prolongate(const FunctionSpace& coarse_space,
                           const std::vector<Real>& coarse_coeffs,
                           const FunctionSpace& fine_space,
                           std::vector<Real>& fine_coeffs,
                           const ReferencePointMap& point_map = {},
                           const ValueTransform& value_transform = {});

    /// Apply restriction from fine to coarse coefficients
    static void restrict_coefficients(const FunctionSpace& fine_space,
                                      const std::vector<Real>& fine_coeffs,
                                      const FunctionSpace& coarse_space,
                                      std::vector<Real>& coarse_coeffs,
                                      const ReferencePointMap& point_map = {},
                                      const ValueTransform& value_transform = {});

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
