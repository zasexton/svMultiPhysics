/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Elements/CompositeElement.h"

#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace elements {

CompositeElement::CompositeElement(std::vector<std::shared_ptr<Element>> components)
    : components_(std::move(components)) {
    if (components_.empty()) {
        throw FEException("CompositeElement requires at least one component",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    primary_ = components_.front();
    if (!primary_) {
        throw FEException("CompositeElement: first component is null",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const ElementType elem_type = primary_->element_type();
    const int dim = primary_->dimension();

    num_dofs_ = 0;
    for (const auto& comp : components_) {
        if (!comp) {
            throw FEException("CompositeElement: encountered null component",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        if (comp->element_type() != elem_type || comp->dimension() != dim) {
            throw FEException("CompositeElement: incompatible component type or dimension",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
        }
        num_dofs_ += comp->num_dofs();
    }

    dimension_ = dim;
    info_.element_type = elem_type;
    info_.field_type   = FieldType::Mixed;
    info_.continuity   = Continuity::Custom;

    int max_order = 0;
    for (const auto& comp : components_) {
        max_order = std::max(max_order, comp->polynomial_order());
    }
    info_.order = max_order;
}

} // namespace elements
} // namespace FE
} // namespace svmp

