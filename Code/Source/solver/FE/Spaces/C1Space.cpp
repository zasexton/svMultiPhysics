/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/C1Space.h"

namespace svmp {
namespace FE {
namespace spaces {

C1Space::C1Space(ElementType element_type,
                 int order)
    : element_type_(element_type),
      polynomial_order_(order) {
    FE_CHECK_ARG(element_type_ == ElementType::Line2 ||
                 element_type_ == ElementType::Quad4 ||
                 element_type_ == ElementType::Hex8,
                 "C1Space is intentionally limited to ElementType::Line2, ElementType::Quad4, and ElementType::Hex8");
    FE_CHECK_ARG(polynomial_order_ == 3,
                 "C1Space is intentionally limited to cubic Hermite order (3)");

    element_dimension_ = element_dimension(element_type_);
    FE_CHECK_ARG(element_dimension_ >= 1 && element_dimension_ <= 3,
                 "C1Space expects a 1D, 2D, or 3D reference element");

    auto basis = std::make_shared<basis::HermiteBasis>(element_type_, polynomial_order_);

    const int quad_order =
        quadrature::QuadratureFactory::recommended_order(basis->order(), true);
    auto quad = quadrature::QuadratureFactory::create(
        element_type_, quad_order, QuadratureType::GaussLegendre, true);

    element_ = std::make_shared<elements::GeneralBasisElement>(
        std::move(basis), std::move(quad), FieldType::Scalar, Continuity::C1);
}

} // namespace spaces
} // namespace FE
} // namespace svmp
