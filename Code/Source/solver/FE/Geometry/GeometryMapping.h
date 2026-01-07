/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_GEOMETRY_GEOMETRYMAPPING_H
#define SVMP_FE_GEOMETRY_GEOMETRYMAPPING_H

/**
 * @file GeometryMapping.h
 * @brief Abstract interface for reference-to-physical element mappings
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Math/Matrix.h"
#include "Math/Vector.h"
#include <array>
#include <vector>

namespace svmp {
namespace FE {
namespace geometry {

/**
 * @brief Base interface for geometric mappings
 *
 * Implementations consume nodal coordinates (typically provided by the Mesh module)
 * but remain mesh-agnostic. All operations are performed in reference space
 * using FE Math types.
 *
 * Notes on Jacobians for embedded geometry (dim < 3):
 * - For dim == 1 (curve) and dim == 2 (surface) mappings in 3D, the returned
 *   Jacobian is a full 3x3 "frame" matrix whose first dim columns are true
 *   tangents (∂x/∂ξ_i) and whose remaining columns form an orthonormal
 *   complement. This makes the Jacobian invertible and enables consistent
 *   inverse mapping and gradient transforms for embedded entities.
 */
class GeometryMapping {
public:
    virtual ~GeometryMapping() = default;

    /// Mapping Hessian type: for each physical component x_m, store d^2 x_m / d xi_i d xi_j
    using MappingHessian = std::array<math::Matrix<Real, 3, 3>, 3>;

    /// Reference element type
    virtual ElementType element_type() const noexcept = 0;

    /// Reference dimension (1, 2, or 3)
    virtual int dimension() const noexcept = 0;

    /// Number of geometry nodes used by the mapping
    virtual std::size_t num_nodes() const noexcept = 0;

    /// Access underlying geometry nodes (for testing/introspection)
    virtual const std::vector<math::Vector<Real, 3>>& nodes() const noexcept = 0;

    /// Map reference coordinates to physical space
    virtual math::Vector<Real, 3> map_to_physical(const math::Vector<Real, 3>& xi) const = 0;

    /// Map physical coordinates to reference space (may use Newton iteration)
    virtual math::Vector<Real, 3> map_to_reference(const math::Vector<Real, 3>& x_phys,
                                                   const math::Vector<Real, 3>& initial_guess = math::Vector<Real, 3>{}) const = 0;

    /// Jacobian matrix d x / d xi at reference point
    virtual math::Matrix<Real, 3, 3> jacobian(const math::Vector<Real, 3>& xi) const = 0;

    /**
     * @brief Second derivatives of the mapping x(xi): d^2 x / d xi^2
     *
     * Returns a 3-component tensor where entry [m](i,j) is:
     *   d^2 x_m / d xi_i d xi_j
     *
     * For lower-dimensional embedded mappings (dimension() < 3), only the leading
     * dim×dim block is meaningful; other entries should be zero. The default
     * implementation returns all zeros (affine/linear mapping).
     */
    virtual MappingHessian mapping_hessian(const math::Vector<Real, 3>& /*xi*/) const { return {}; }

    /// Inverse Jacobian at reference point (throws if singular)
    virtual math::Matrix<Real, 3, 3> jacobian_inverse(const math::Vector<Real, 3>& xi) const;

    /// Jacobian determinant at reference point
    virtual Real jacobian_determinant(const math::Vector<Real, 3>& xi) const;

    /// Convenience: transform a reference gradient to physical space (J^{-T} * grad)
    math::Vector<Real, 3> transform_gradient(const math::Vector<Real, 3>& grad_ref,
                                             const math::Vector<Real, 3>& xi) const;
};

// -----------------------------------------------------------------------------
// Inline helpers
// -----------------------------------------------------------------------------

inline math::Matrix<Real, 3, 3> GeometryMapping::jacobian_inverse(const math::Vector<Real, 3>& xi) const {
    return jacobian(xi).inverse();
}

inline Real GeometryMapping::jacobian_determinant(const math::Vector<Real, 3>& xi) const {
    return jacobian(xi).determinant();
}

inline math::Vector<Real, 3> GeometryMapping::transform_gradient(const math::Vector<Real, 3>& grad_ref,
                                                                 const math::Vector<Real, 3>& xi) const {
    auto JinvT = jacobian_inverse(xi).transpose();
    math::Vector<Real, 3> g = grad_ref;
    const std::size_t dim = static_cast<std::size_t>(dimension());
    for (std::size_t k = dim; k < 3; ++k) {
        g[k] = Real(0);
    }

    math::Vector<Real, 3> out{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            out[i] += JinvT(i, j) * g[j];
        }
    }
    return out;
}

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_GEOMETRYMAPPING_H
