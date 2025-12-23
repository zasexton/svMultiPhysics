/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_L2SPACE_H
#define SVMP_FE_SPACES_L2SPACE_H

/**
 * @file L2Space.h
 * @brief L²-discontinuous scalar function space
 */

#include "Spaces/FunctionSpace.h"
#include "Elements/DiscontinuousElement.h"

namespace svmp {
namespace FE {
namespace spaces {

/**
 * @brief L²-discontinuous space based on scalar DG elements
 *
 * This space represents cell-wise discontinuous scalar fields typically used
 * in DG formulations. Continuity is classified as L2, and each element owns
 * its own set of DOFs.
 */
class L2Space : public FunctionSpace {
public:
    L2Space(ElementType element_type,
            int order);

    SpaceType space_type() const noexcept override { return SpaceType::L2; }
    FieldType field_type() const noexcept override { return FieldType::Scalar; }
    Continuity continuity() const noexcept override { return Continuity::L2; }

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

#endif // SVMP_FE_SPACES_L2SPACE_H

