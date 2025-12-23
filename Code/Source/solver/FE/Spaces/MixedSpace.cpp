/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/MixedSpace.h"

namespace svmp {
namespace FE {
namespace spaces {

void MixedSpace::add_component(const std::string& name,
                               std::shared_ptr<FunctionSpace> space) {
    FE_CHECK_NOT_NULL(space.get(), "MixedSpace::add_component space");
    components_.push_back(Component{name, std::move(space)});
}

FunctionSpace::Value MixedSpace::evaluate_component(
    std::size_t component_index,
    const Value& xi,
    const std::vector<Real>& coefficients) const {
    if (component_index >= components_.size()) {
        FE_THROW(InvalidArgumentException, "MixedSpace::evaluate_component: invalid component index");
    }
    FE_CHECK_ARG(coefficients.size() == dofs_per_element(),
                 "MixedSpace::evaluate_component: coefficient size mismatch");

    const auto& comp = components_[component_index];
    FE_CHECK_NOT_NULL(comp.space.get(), "MixedSpace::evaluate_component space");

    const std::size_t offset = component_offset(component_index);
    const std::size_t ndofs = comp.space->dofs_per_element();

    std::vector<Real> local_coeffs;
    local_coeffs.reserve(ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        local_coeffs.push_back(coefficients[offset + i]);
    }

    return comp.space->evaluate(xi, local_coeffs);
}

std::vector<FunctionSpace::Value> MixedSpace::evaluate_components(
    const Value& xi,
    const std::vector<Real>& coefficients) const {
    FE_CHECK_ARG(coefficients.size() == dofs_per_element(),
                 "MixedSpace::evaluate_components: coefficient size mismatch");

    std::vector<Value> values;
    values.reserve(components_.size());
    for (std::size_t c = 0; c < components_.size(); ++c) {
        values.push_back(evaluate_component(c, xi, coefficients));
    }
    return values;
}

std::size_t MixedSpace::dofs_per_element() const noexcept {
    std::size_t total = 0;
    for (const auto& c : components_) {
        if (c.space) {
            total += c.space->dofs_per_element();
        }
    }
    return total;
}

std::size_t MixedSpace::component_offset(std::size_t i) const {
    if (i >= components_.size()) {
        FE_THROW(InvalidArgumentException, "MixedSpace::component_offset: invalid component index");
    }
    std::size_t offset = 0;
    for (std::size_t k = 0; k < i; ++k) {
        offset += components_[k].space->dofs_per_element();
    }
    return offset;
}

int MixedSpace::value_dimension() const noexcept {
    int dim = 0;
    for (const auto& c : components_) {
        if (c.space) {
            dim += c.space->value_dimension();
        }
    }
    return dim;
}

int MixedSpace::topological_dimension() const noexcept {
    if (components_.empty() || !components_.front().space) {
        return -1;
    }
    return components_.front().space->topological_dimension();
}

int MixedSpace::polynomial_order() const noexcept {
    int max_order = 0;
    for (const auto& c : components_) {
        if (c.space) {
            max_order = std::max(max_order, c.space->polynomial_order());
        }
    }
    return max_order;
}

ElementType MixedSpace::element_type() const noexcept {
    if (components_.empty() || !components_.front().space) {
        return ElementType::Unknown;
    }
    return components_.front().space->element_type();
}

const elements::Element& MixedSpace::element() const noexcept {
    return components_.front().space->element();
}

std::shared_ptr<const elements::Element> MixedSpace::element_ptr() const noexcept {
    if (components_.empty() || !components_.front().space) {
        return {};
    }
    return components_.front().space->element_ptr();
}

} // namespace spaces
} // namespace FE
} // namespace svmp
