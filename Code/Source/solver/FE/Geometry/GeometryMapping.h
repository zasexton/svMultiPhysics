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
#include <cstdint>
#include <vector>

namespace svmp {
namespace FE {

namespace basis { class BasisFunction; }

namespace geometry {

/**
 * @brief Invert a 3x3 Jacobian after removing its physical column scales.
 *
 * The normalized determinant check is invariant under positive rescaling of
 * the Jacobian columns.  It therefore rejects singular and near-collinear
 * frames without rejecting a uniformly small, shape-regular physical cell
 * solely because its determinant is small in the chosen coordinate units.
 */
[[nodiscard]] math::Matrix<Real, 3, 3> scaleConditionedJacobianInverse(
    const math::Matrix<Real, 3, 3>& jacobian);

/// Return the signed physical determinant after checking column directions for degeneracy.
/// The optional inverse uses the same normalized columns.
[[nodiscard]] Real checkedJacobianDeterminant(const math::Matrix<Real, 3, 3>& jacobian,
                                              math::Matrix<Real, 3, 3>* inverse = nullptr);

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

    /// True when the mapping Jacobian is constant on the reference element.
    virtual bool isAffine() const noexcept { return false; }

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

    /// Replace the nodal coordinates for reuse across cells of the same element type.
    virtual void resetNodes(const std::vector<math::Vector<Real, 3>>& nodes) {
        (void)nodes;
        FE_THROW(FEException, "resetNodes not supported for this mapping type");
    }

    /// Access the underlying geometry basis function (for BasisCache lookup).
    virtual const basis::BasisFunction& geometryBasis() const {
        FE_THROW(FEException, "geometryBasis not supported for this mapping type");
    }

    /// Convenience: transform a reference gradient to physical space (J^{-T} * grad)
    math::Vector<Real, 3> transform_gradient(const math::Vector<Real, 3>& grad_ref,
                                             const math::Vector<Real, 3>& xi) const;

    /// Convenience: transform a reference gradient using a precomputed inverse Jacobian.
    math::Vector<Real, 3> transform_gradient(const math::Vector<Real, 3>& grad_ref,
                                             const math::Matrix<Real, 3, 3>& jacobian_inverse) const;

protected:
    GeometryMapping() = default;
    // Copies and moved objects receive independent cache generations on first use.
    GeometryMapping(const GeometryMapping&) noexcept {}
    GeometryMapping(GeometryMapping&& other) noexcept { other.invalidateJacobianCache(); }
    GeometryMapping& operator=(const GeometryMapping&) noexcept {
        invalidateJacobianCache();
        return *this;
    }
    GeometryMapping& operator=(GeometryMapping&& other) noexcept {
        invalidateJacobianCache();
        other.invalidateJacobianCache();
        return *this;
    }

    /// Call when changing a mapping parameter that is not represented by nodes().
    void invalidateJacobianCache() noexcept { jacobian_cache_generation_ = 0; }

private:
    friend class JacobianCache;
    // Issued under the cache mutex. Uncached element mappings incur no synchronization.
    mutable std::uint64_t jacobian_cache_generation_{0};
};

// -----------------------------------------------------------------------------
// Inline helpers
// -----------------------------------------------------------------------------

inline math::Matrix<Real, 3, 3> GeometryMapping::jacobian_inverse(const math::Vector<Real, 3>& xi) const {
    return scaleConditionedJacobianInverse(jacobian(xi));
}

inline Real GeometryMapping::jacobian_determinant(const math::Vector<Real, 3>& xi) const {
    return jacobian(xi).determinant();
}

inline math::Vector<Real, 3> GeometryMapping::transform_gradient(const math::Vector<Real, 3>& grad_ref,
                                                                 const math::Vector<Real, 3>& xi) const {
    return transform_gradient(grad_ref, jacobian_inverse(xi));
}

inline math::Vector<Real, 3> GeometryMapping::transform_gradient(const math::Vector<Real, 3>& grad_ref,
                                                                 const math::Matrix<Real, 3, 3>& jacobian_inverse) const {
    auto JinvT = jacobian_inverse.transpose();
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
