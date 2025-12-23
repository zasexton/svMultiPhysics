/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Elements/MixedElement.h"

#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace elements {

MixedElement::MixedElement(std::vector<MixedSubElement> sub_elements)
    : sub_elements_(std::move(sub_elements)) {
    if (sub_elements_.empty()) {
        throw FEException("MixedElement requires at least one sub-element",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    primary_ = sub_elements_.front().element;
    if (!primary_) {
        throw FEException("MixedElement: first sub-element is null",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // All sub-elements must agree on element type and dimension for a
    // well-defined reference element.
    const ElementType elem_type = primary_->element_type();
    const int dim = primary_->dimension();

    num_dofs_ = 0;
    for (const auto& sub : sub_elements_) {
        if (!sub.element) {
            throw FEException("MixedElement: encountered null sub-element",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        if (sub.element->element_type() != elem_type ||
            sub.element->dimension() != dim) {
            throw FEException("MixedElement: incompatible sub-element type or dimension",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
        }
        num_dofs_ += sub.element->num_dofs();
    }

    dimension_ = dim;
    info_.element_type = elem_type;
    info_.field_type   = FieldType::Mixed;
    info_.continuity   = Continuity::Custom;

    // For informational purposes we record the maximum polynomial order
    // among all sub-elements.
    int max_order = 0;
    for (const auto& sub : sub_elements_) {
        max_order = std::max(max_order, sub.element->polynomial_order());
    }
    info_.order = max_order;
}

} // namespace elements
} // namespace FE
} // namespace svmp

