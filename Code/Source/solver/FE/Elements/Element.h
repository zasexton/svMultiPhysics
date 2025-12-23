/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ELEMENTS_ELEMENT_H
#define SVMP_FE_ELEMENTS_ELEMENT_H

/**
 * @file Element.h
 * @brief Abstract interface for finite elements
 *
 * The Element interface represents a reference finite element definition that
 * combines:
 *  - a reference element type (`ElementType`)
 *  - a basis function (`BasisFunction`)
 *  - a quadrature rule (`QuadratureRule`)
 *
 * It is intentionally independent of any particular mesh instance. Geometry
 * (physical coordinates of nodes) is handled by the Geometry module via
 * `GeometryMapping`. Higher-level modules (Dofs, Spaces, Assembly) use this
 * interface to reason about element-local degrees of freedom without
 * depending on the Mesh library.
 */

#include "Core/Types.h"
#include "Basis/BasisFunction.h"
#include "Quadrature/QuadratureRule.h"

#include <memory>

namespace svmp {
namespace FE {
namespace elements {

/**
 * @brief Lightweight descriptor of an element's discrete characteristics
 */
struct ElementInfo {
    ElementType element_type{ElementType::Unknown};
    FieldType   field_type{FieldType::Scalar};
    Continuity  continuity{Continuity::C0};
    int         order{1};
};

/**
 * @brief Abstract base class for all finite element types
 *
 * Concrete element classes (Lagrange, DG, vector-valued, spectral, mixed,
 * etc.) implement this interface. The goal is to provide enough information
 * for function-space and assembly modules while keeping the interface small
 * and mesh-agnostic.
 */
class Element {
public:
    virtual ~Element() = default;

    /// Element metadata (type, field kind, continuity, polynomial order)
    virtual ElementInfo info() const noexcept = 0;

    /// Convenience accessors
    ElementType element_type() const noexcept { return info().element_type; }
    FieldType   field_type()   const noexcept { return info().field_type; }
    Continuity  continuity()   const noexcept { return info().continuity; }
    int         polynomial_order() const noexcept { return info().order; }

    /// Topological dimension of the reference element (1, 2, or 3)
    virtual int dimension() const noexcept = 0;

    /**
     * @brief Number of degrees of freedom carried by this element
     *
     * By definition in this library, each basis function corresponds to one
     * degree of freedom, regardless of whether it is scalar- or vector-valued.
     * For scalar H¹ elements this equals the number of scalar shape functions.
     * For H(div)/H(curl) elements, this equals the number of vector-valued
     * basis functions (one DOF per flux/edge functional in the classical sense).
     */
    virtual std::size_t num_dofs() const noexcept = 0;

    /**
     * @brief Number of basis functions (shape functions)
     *
     * For scalar nodes this matches the number of nodal points. For vector
     * elements it is the number of vector-valued basis functions.
     */
    virtual std::size_t num_nodes() const noexcept = 0;

    /// Access underlying basis (non-owning reference)
    virtual const basis::BasisFunction& basis() const noexcept = 0;

    /// Shared pointer to basis (for lifetime management / factories)
    virtual std::shared_ptr<const basis::BasisFunction> basis_ptr() const noexcept = 0;

    /// Quadrature rule associated with this element
    virtual std::shared_ptr<const quadrature::QuadratureRule> quadrature() const noexcept = 0;
};

} // namespace elements
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ELEMENTS_ELEMENT_H
