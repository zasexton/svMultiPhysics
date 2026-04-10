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
#include "Elements/ElementFactory.h"

namespace svmp {
namespace FE {
namespace spaces {

/**
 * @brief L²-discontinuous scalar function space
 *
 * This space represents cell-wise discontinuous scalar fields typically used
 * in DG formulations. It wraps scalar elements compatible with L2 continuity,
 * including nodal DG elements and alternative scalar basis families exposed
 * through ElementFactory.
 */
class L2Space : public FunctionSpace {
public:
    L2Space(ElementType element_type,
            int order);

    L2Space(ElementType element_type,
            int order,
            BasisType basis_type);

    explicit L2Space(const elements::ElementRequest& request);

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
