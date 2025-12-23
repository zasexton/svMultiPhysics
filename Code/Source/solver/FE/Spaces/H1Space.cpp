/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/H1Space.h"

namespace svmp {
namespace FE {
namespace spaces {

H1Space::H1Space(ElementType element_type,
                 int order)
    : element_type_(element_type),
      order_(order) {
    FE_CHECK_ARG(order_ >= 0, "H1Space requires non-negative polynomial order");

    dimension_ = element_dimension(element_type_);
    FE_CHECK_ARG(dimension_ >= 0, "H1Space: unknown element dimension");

    element_ = std::make_shared<elements::LagrangeElement>(
        element_type_, order_, FieldType::Scalar, Continuity::C0);
}

} // namespace spaces
} // namespace FE
} // namespace svmp

