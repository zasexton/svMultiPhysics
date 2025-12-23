/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_H1SPACE_H
#define SVMP_FE_SPACES_H1SPACE_H

/**
 * @file H1Space.h
 * @brief Standard H¹-conforming (C0) scalar function space
 */

#include "Spaces/FunctionSpace.h"
#include "Elements/LagrangeElement.h"

namespace svmp {
namespace FE {
namespace spaces {

/**
 * @brief H¹-conforming scalar function space built from Lagrange elements
 *
 * This space represents standard C0 continuous finite element spaces used for
 * scalar fields such as pressure or temperature. It uses @ref elements::LagrangeElement
 * as the underlying prototype element.
 */
class H1Space : public FunctionSpace {
public:
    H1Space(ElementType element_type,
            int order);

    SpaceType space_type() const noexcept override { return SpaceType::H1; }
    FieldType field_type() const noexcept override { return FieldType::Scalar; }
    Continuity continuity() const noexcept override { return Continuity::C0; }

    int value_dimension() const noexcept override { return 1; }
    int topological_dimension() const noexcept override { return dimension_; }
    int polynomial_order() const noexcept override { return order_; }
    ElementType element_type() const noexcept override { return element_type_; }

    const elements::Element& element() const noexcept override { return *element_; }
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override { return element_; }

private:
    ElementType element_type_;
    int order_;
    int dimension_;
    std::shared_ptr<elements::Element> element_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_H1SPACE_H

