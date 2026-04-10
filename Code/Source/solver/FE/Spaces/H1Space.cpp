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

namespace {

elements::ElementRequest make_h1_request(ElementType element_type,
                                         int order,
                                         BasisType basis_type) {
    elements::ElementRequest req;
    req.element_type = element_type;
    req.basis_type = basis_type;
    req.field_type = FieldType::Scalar;
    req.continuity = Continuity::C0;
    req.order = order;
    return req;
}

elements::ElementRequest normalize_h1_request(const elements::ElementRequest& request) {
    FE_THROW_IF(request.element_type == ElementType::Unknown, InvalidArgumentException,
                "H1Space requires a concrete element type");
    FE_THROW_IF(request.field_type != FieldType::Scalar, InvalidArgumentException,
                "H1Space requires FieldType::Scalar");
    FE_THROW_IF(request.continuity != Continuity::C0, InvalidArgumentException,
                "H1Space requires Continuity::C0");
    return request;
}

} // namespace

H1Space::H1Space(ElementType element_type,
                 int order)
    : H1Space(element_type, order, BasisType::Lagrange) {}

H1Space::H1Space(ElementType element_type,
                 int order,
                 BasisType basis_type)
    : H1Space(make_h1_request(element_type, order, basis_type)) {}

H1Space::H1Space(const elements::ElementRequest& request) {
    const auto req = normalize_h1_request(request);
    if (req.order.has_value()) {
        FE_CHECK_ARG(*req.order >= 0, "H1Space requires non-negative polynomial order");
    }

    element_ = elements::ElementFactory::create(req);
    FE_CHECK_NOT_NULL(element_.get(), "H1Space element");

    element_type_ = element_->element_type();
    order_ = element_->polynomial_order();
    dimension_ = element_->dimension();
    FE_CHECK_ARG(dimension_ >= 0, "H1Space: unknown element dimension");
    FE_THROW_IF(element_->field_type() != FieldType::Scalar, InvalidArgumentException,
                "H1Space requires a scalar element");
    FE_THROW_IF(element_->continuity() != Continuity::C0, InvalidArgumentException,
                "H1Space requires a C0-conforming element");
}

} // namespace spaces
} // namespace FE
} // namespace svmp
