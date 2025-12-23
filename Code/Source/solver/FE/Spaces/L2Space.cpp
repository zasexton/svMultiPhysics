/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/L2Space.h"

namespace svmp {
namespace FE {
namespace spaces {

L2Space::L2Space(ElementType element_type,
                 int order)
    : element_type_(element_type),
      order_(order) {
    FE_CHECK_ARG(order_ >= 0, "L2Space requires non-negative polynomial order");

    dimension_ = element_dimension(element_type_);
    FE_CHECK_ARG(dimension_ >= 0, "L2Space: unknown element dimension");

    element_ = std::make_shared<elements::DiscontinuousElement>(
        element_type_, order_, FieldType::Scalar);
}

} // namespace spaces
} // namespace FE
} // namespace svmp

