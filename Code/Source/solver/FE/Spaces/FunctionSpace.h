/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_FUNCTIONSPACE_H
#define SVMP_FE_SPACES_FUNCTIONSPACE_H

/**
 * @file FunctionSpace.h
 * @brief Abstract interface for finite element function spaces
 *
 * The Spaces module defines function spaces on top of reference finite
 * elements provided by the Elements module. Function spaces are pure FE
 * abstractions – they do not depend on any particular Mesh data structure
 * and operate entirely in reference coordinates using FE Math, Basis, and
 * Quadrature utilities.
 *
 * A FunctionSpace describes:
 *  - the underlying finite element (ElementType, polynomial order,
 *    continuity, field type),
 *  - the number of local degrees of freedom per element,
 *  - basic interpolation and evaluation routines on the reference element.
 *
 * Global topology (which elements exist, how they are connected) and global
 * DOF numbering are owned by the Mesh and Dofs modules respectively.  This
 * header intentionally avoids any direct dependency on the Mesh library.
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Elements/Element.h"
#include "Math/Vector.h"
#include <functional>
#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace spaces {

/// High-level family of function space
enum class SpaceType : std::uint8_t {
    H1,
    C1,
    L2,
    HCurl,
    HDiv,
    Mixed,
    Product,
    Trace,
    Composite,
    Enriched,
    Adaptive,
    Isogeometric
};

/**
 * @brief Base interface for all function spaces
 *
 * Function spaces in this library are element-local abstractions that know
 * how to interpolate and evaluate fields on a single reference element using
 * the underlying basis and quadrature associated with an Elements::Element.
 *
 * The default implementations of @ref interpolate and @ref evaluate perform
 * an L² projection on the reference element using the element's quadrature
 * rule. This provides a robust, mesh-agnostic interpolation mechanism that
 * works for scalar H¹/L² spaces as well as vector-valued H(div)/H(curl)
 * spaces. Composite spaces (mixed, enriched, etc.) may override these
 * methods to provide custom behavior or to disallow unsupported operations.
 */
class FunctionSpace {
public:
    using Value      = math::Vector<Real, 3>;  ///< Field value (scalar in first component for scalar spaces)
    using Gradient   = math::Vector<Real, 3>;  ///< Gradient in reference coordinates
    using ValueFunction = std::function<Value(const Value&)>; ///< f(x) callback in reference coordinates

    virtual ~FunctionSpace() = default;

    /// Space family identifier
    virtual SpaceType space_type() const noexcept = 0;

    /// Field type (scalar, vector, tensor, mixed)
    virtual FieldType field_type() const noexcept = 0;

    /// Continuity classification
    virtual Continuity continuity() const noexcept = 0;

    /// Number of value components (1 for scalar fields, 2/3 for vectors)
    virtual int value_dimension() const noexcept = 0;

    /// Topological dimension of the reference element
    virtual int topological_dimension() const noexcept = 0;

    /// Polynomial order in the sense of the underlying basis
    virtual int polynomial_order() const noexcept = 0;

    /// Underlying reference element type
    virtual ElementType element_type() const noexcept = 0;

    /// Access to underlying prototype element (non-owning reference)
    virtual const elements::Element& element() const noexcept = 0;

    /**
     * @brief Compatibility accessor for cell-dependent element queries
     *
     * Some assembly paths expect a `getElement(cell_type, cell_id)` API even
     * for spaces that use a single prototype element. The default
     * implementation ignores the arguments and returns @ref element().
     */
    virtual const elements::Element& getElement(ElementType /*cell_type*/,
                                                GlobalIndex /*cell_id*/) const noexcept {
        return element();
    }

    /// Shared pointer to underlying prototype element
    virtual std::shared_ptr<const elements::Element> element_ptr() const noexcept = 0;

    /// Number of local DOFs per element (one scalar DOF per basis function)
    virtual std::size_t dofs_per_element() const noexcept {
        return element().num_dofs();
    }

    /**
     * @brief Interpolate a reference-space function into this space
     *
     * Computes coefficients @p coefficients such that the finite element
     * approximation u_h best matches the target function in the L² sense:
     *     ∫ (u_h - f) · v dx = 0  for all basis functions v.
     *
     * Integration is carried out on the reference element using the
     * quadrature rule associated with the underlying Element. For scalar
     * spaces the scalar value is stored in the first component of Value.
     *
     * @param function Target function f(x̂) expressed in reference coordinates
     * @param[out] coefficients Output vector resized to dofs_per_element()
     *
     * @throws FEException if the mass matrix is singular or ill-conditioned
     */
    virtual void interpolate(const ValueFunction& function,
                             std::vector<Real>& coefficients) const;

    /**
     * @brief Convenience scalar interpolation wrapper
     *
     * Treats the provided scalar function as a vector-valued function with
     * value in the first component and zeros elsewhere.
     */
    void interpolate_scalar(const std::function<Real(const Value&)>& function,
                            std::vector<Real>& coefficients) const;

    /**
     * @brief Evaluate finite element field at reference point
     *
     * Given local DOF coefficients (one scalar per basis function), evaluate
     * the finite element approximation at point @p xi using the underlying
     * basis. For scalar spaces the scalar value is returned in the first
     * component of the result and remaining components are zero.
     *
     * @param xi Reference coordinates in [-1,1]^d (unused components ignored)
     * @param coefficients Local DOF coefficients (size must equal dofs_per_element())
     * @return Field value as a 3D vector (scalar in first slot for scalar fields)
     */
    virtual Value evaluate(const Value& xi,
                           const std::vector<Real>& coefficients) const;

    /**
     * @brief Evaluate reference-space gradient at a point
     *
     * Computes ∇u_h in reference coordinates. For scalar spaces this is the
     * gradient of the scalar field. For vector-valued spaces, use the
     * appropriate operators (e.g., curl/div) instead.
     *
     * @param xi Reference coordinates (unused components ignored)
     * @param coefficients Local DOF coefficients (size must equal dofs_per_element())
     * @return Gradient vector in reference coordinates
     */
    virtual Gradient evaluate_gradient(const Value& xi,
                                       const std::vector<Real>& coefficients) const;

    /**
     * @brief Evaluate divergence of a vector-valued field at a point
     *
     * @throws FEException if called on a scalar-valued basis.
     */
    virtual Real evaluate_divergence(const Value& xi,
                                     const std::vector<Real>& coefficients) const;

    /// Alias for divergence (common shorthand)
    Real evaluate_div(const Value& xi,
                      const std::vector<Real>& coefficients) const {
        return evaluate_divergence(xi, coefficients);
    }

    /**
     * @brief Evaluate curl of a vector-valued field at a point
     *
     * For 2D vector bases, curl is returned in the z-component.
     *
     * @throws FEException if called on a scalar-valued basis.
     */
    virtual Value evaluate_curl(const Value& xi,
                                const std::vector<Real>& coefficients) const;

    /// Convenience wrapper for scalar fields (returns first component only)
    Real evaluate_scalar(const Value& xi,
                         const std::vector<Real>& coefficients) const {
        return evaluate(xi, coefficients)[0];
    }
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_FUNCTIONSPACE_H
