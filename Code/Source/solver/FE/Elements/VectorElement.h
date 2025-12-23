/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ELEMENTS_VECTORELEMENT_H
#define SVMP_FE_ELEMENTS_VECTORELEMENT_H

/**
 * @file VectorElement.h
 * @brief H(div)/H(curl) vector-valued finite elements
 */

#include "Elements/Element.h"
#include "Basis/BasisFactory.h"
#include "Quadrature/QuadratureFactory.h"

namespace svmp {
namespace FE {
namespace elements {

/**
 * @brief Vector-valued conforming element for H(div) or H(curl) spaces
 *
 * Uses Raviart-Thomas, Nedelec, or BDM bases as provided by BasisFactory
 * depending on the requested continuity. The field type is always Vector.
 */
class VectorElement : public Element {
public:
    /// Construct an H(div) or H(curl) vector element
    VectorElement(ElementType element_type,
                  int order,
                  Continuity continuity);

    /// Construct an H(div) or H(curl) vector element with an explicit vector basis family
    VectorElement(ElementType element_type,
                  int order,
                  Continuity continuity,
                  BasisType basis_type);

    ElementInfo info() const noexcept override { return info_; }
    int dimension() const noexcept override { return dimension_; }

    std::size_t num_dofs() const noexcept override { return num_dofs_; }
    std::size_t num_nodes() const noexcept override { return basis_->size(); }

    const basis::BasisFunction& basis() const noexcept override { return *basis_; }
    std::shared_ptr<const basis::BasisFunction> basis_ptr() const noexcept override { return basis_; }

    std::shared_ptr<const quadrature::QuadratureRule> quadrature() const noexcept override { return quad_; }

private:
    ElementInfo info_;
    int dimension_;
    std::size_t num_dofs_;
    BasisType basis_type_{BasisType::Lagrange};
    std::shared_ptr<basis::BasisFunction> basis_;
    std::shared_ptr<const quadrature::QuadratureRule> quad_;
};

} // namespace elements
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ELEMENTS_VECTORELEMENT_H
