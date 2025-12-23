/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ELEMENTS_SPECTRALELEMENT_H
#define SVMP_FE_ELEMENTS_SPECTRALELEMENT_H

/**
 * @file SpectralElement.h
 * @brief High-order spectral elements based on Gauss-Lobatto nodes
 */

#include "Elements/Element.h"
#include "Basis/BasisFactory.h"
#include "Quadrature/QuadratureFactory.h"

namespace svmp {
namespace FE {
namespace elements {

/**
 * @brief Scalar spectral element using `basis::SpectralBasis`
 *
 * Uses Gauss-Lobatto quadrature by default to align quadrature points with
 * spectral nodal locations on tensor-product reference elements.
 */
class SpectralElement : public Element {
public:
    SpectralElement(ElementType element_type,
                    int order,
                    FieldType field_type = FieldType::Scalar);

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

#endif // SVMP_FE_ELEMENTS_SPECTRALELEMENT_H

