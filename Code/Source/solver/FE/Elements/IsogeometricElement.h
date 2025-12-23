/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ELEMENTS_ISOGEOMETRICELEMENT_H
#define SVMP_FE_ELEMENTS_ISOGEOMETRICELEMENT_H

/**
 * @file IsogeometricElement.h
 * @brief Generic isogeometric (IGA) element wrapper
 *
 * This element is designed to host NURBS or B-spline bases that implement
 * the `BasisFunction` interface. The Element layer does not implement NURBS
 * itself; instead, external code can provide a suitable basis and quadrature
 * rule and wrap them inside an `IsogeometricElement`.
 */

#include "Elements/Element.h"

namespace svmp {
namespace FE {
namespace elements {

class IsogeometricElement : public Element {
public:
    IsogeometricElement(std::shared_ptr<basis::BasisFunction> basis,
                        std::shared_ptr<const quadrature::QuadratureRule> quadrature,
                        FieldType field_type,
                        Continuity continuity);

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
    std::shared_ptr<basis::BasisFunction> basis_;
    std::shared_ptr<const quadrature::QuadratureRule> quad_;
};

} // namespace elements
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ELEMENTS_ISOGEOMETRICELEMENT_H

