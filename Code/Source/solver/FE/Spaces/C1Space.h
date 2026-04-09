/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_C1SPACE_H
#define SVMP_FE_SPACES_C1SPACE_H

/**
 * @file C1Space.h
 * @brief C¹-continuous scalar function space on Hermite-supported cells
 *
 * This space provides a C¹ function space built from the cubic Hermite basis
 * on every topology currently supported by `HermiteBasis`: line, quadrilateral,
 * and hexahedral tensor-product cells.
 *
 * The implementation is mesh-agnostic and relies only on the Elements, Basis,
 * and Quadrature modules. It uses elements::GeneralBasisElement internally as
 * a generic basis-backed element host with Hermite basis and Gauss-Legendre
 * quadrature.
 */

#include "Spaces/FunctionSpace.h"
#include "Elements/GeneralBasisElement.h"
#include "Basis/HermiteBasis.h"
#include "Quadrature/QuadratureFactory.h"

namespace svmp {
namespace FE {
namespace spaces {

/**
 * @brief C¹-continuous scalar space on Hermite-supported elements
 */
class C1Space : public FunctionSpace {
public:
    /// Construct C¹ space on Line2, Quad4, or Hex8 using cubic Hermite basis
    C1Space(ElementType element_type,
            int order = 3);

    SpaceType space_type() const noexcept override { return SpaceType::C1; }
    FieldType field_type() const noexcept override { return FieldType::Scalar; }
    Continuity continuity() const noexcept override { return Continuity::C1; }

    int value_dimension() const noexcept override { return 1; }
    int topological_dimension() const noexcept override { return element_dimension_; }
    int polynomial_order() const noexcept override { return polynomial_order_; }
    ElementType element_type() const noexcept override { return element_type_; }

    const elements::Element& element() const noexcept override { return *element_; }
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override { return element_; }

private:
    ElementType element_type_{ElementType::Unknown};
    int element_dimension_{0};
    int polynomial_order_{3};
    std::shared_ptr<elements::Element> element_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_C1SPACE_H
