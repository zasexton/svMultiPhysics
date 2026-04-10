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

namespace {

elements::ElementRequest make_l2_request(ElementType element_type,
                                         int order,
                                         BasisType basis_type) {
    elements::ElementRequest req;
    req.element_type = element_type;
    req.basis_type = basis_type;
    req.field_type = FieldType::Scalar;
    req.continuity = Continuity::L2;
    req.order = order;
    return req;
}

elements::ElementRequest normalize_l2_request(const elements::ElementRequest& request) {
    FE_THROW_IF(request.element_type == ElementType::Unknown, InvalidArgumentException,
                "L2Space requires a concrete element type");
    FE_THROW_IF(request.field_type != FieldType::Scalar, InvalidArgumentException,
                "L2Space requires FieldType::Scalar");
    FE_THROW_IF(request.continuity != Continuity::L2, InvalidArgumentException,
                "L2Space requires Continuity::L2");
    return request;
}

} // namespace

L2Space::L2Space(ElementType element_type,
                 int order)
    : L2Space(element_type, order, BasisType::Lagrange) {}

L2Space::L2Space(ElementType element_type,
                 int order,
                 BasisType basis_type)
    : L2Space(make_l2_request(element_type, order, basis_type)) {}

L2Space::L2Space(const elements::ElementRequest& request) {
    const auto req = normalize_l2_request(request);
    if (req.order.has_value()) {
        FE_CHECK_ARG(*req.order >= 0, "L2Space requires non-negative polynomial order");
    }

    element_ = elements::ElementFactory::create(req);
    FE_CHECK_NOT_NULL(element_.get(), "L2Space element");

    element_type_ = element_->element_type();
    order_ = element_->polynomial_order();
    dimension_ = element_->dimension();
    FE_CHECK_ARG(dimension_ >= 0, "L2Space: unknown element dimension");
    FE_THROW_IF(element_->field_type() != FieldType::Scalar, InvalidArgumentException,
                "L2Space requires a scalar element");
    FE_THROW_IF(element_->continuity() != Continuity::L2, InvalidArgumentException,
                "L2Space requires an L2-discontinuous element");
}

} // namespace spaces
} // namespace FE
} // namespace svmp
