/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ELEMENTS_GENERALBASISELEMENT_H
#define SVMP_FE_ELEMENTS_GENERALBASISELEMENT_H

/**
 * @file GeneralBasisElement.h
 * @brief Generic basis-backed scalar/vector element wrapper
 *
 * This class is a thin Element host for any externally constructed
 * `BasisFunction` plus compatible quadrature rule. The Element layer does not
 * implement the basis itself; it owns only the FE metadata and the
 * basis/quadrature pairing.
 */

#include "Elements/Element.h"

namespace svmp {
namespace FE {
namespace elements {

class GeneralBasisElement : public Element {
public:
    /// Construct a generic basis-backed element from externally supplied basis/quadrature
    GeneralBasisElement(std::shared_ptr<basis::BasisFunction> basis,
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

#endif // SVMP_FE_ELEMENTS_GENERALBASISELEMENT_H
