/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/AdaptiveSpace.h"

namespace svmp {
namespace FE {
namespace spaces {

void AdaptiveSpace::add_level(int order,
                              std::shared_ptr<FunctionSpace> space) {
    FE_CHECK_NOT_NULL(space.get(), "AdaptiveSpace::add_level space");
    FE_CHECK_ARG(order >= 0, "AdaptiveSpace::add_level: order must be non-negative");

    // Ensure consistent element type and field type
    if (!levels_.empty()) {
        const auto& ref = levels_.front();
        FE_CHECK_ARG(space->element_type() == ref.space->element_type(),
                     "AdaptiveSpace: all levels must share the same element type");
        FE_CHECK_ARG(space->field_type() == ref.space->field_type(),
                     "AdaptiveSpace: all levels must share the same field type");
    }

    levels_.push_back(Level{order, std::move(space)});

    // Keep active index pointing to the highest order by default
    std::size_t max_idx = 0;
    int max_order = levels_[0].order;
    for (std::size_t i = 1; i < levels_.size(); ++i) {
        if (levels_[i].order > max_order) {
            max_order = levels_[i].order;
            max_idx = i;
        }
    }
    active_index_ = max_idx;
}

void AdaptiveSpace::set_active_level(std::size_t index) {
    FE_CHECK_ARG(index < levels_.size(), "AdaptiveSpace::set_active_level: index out of range");
    active_index_ = index;
}

void AdaptiveSpace::set_active_level_by_order(int order) {
    for (std::size_t i = 0; i < levels_.size(); ++i) {
        if (levels_[i].order == order) {
            active_index_ = i;
            return;
        }
    }
    FE_THROW(InvalidArgumentException,
             "AdaptiveSpace::set_active_level_by_order: requested order not found");
}

const AdaptiveSpace::Level& AdaptiveSpace::active_level() const {
    FE_CHECK_ARG(!levels_.empty(), "AdaptiveSpace::active_level: no levels defined");
    return levels_[active_index_];
}

FieldType AdaptiveSpace::field_type() const noexcept {
    if (levels_.empty() || !levels_.front().space) {
        return FieldType::Scalar;
    }
    return levels_.front().space->field_type();
}

Continuity AdaptiveSpace::continuity() const noexcept {
    if (levels_.empty() || !levels_.front().space) {
        return Continuity::Custom;
    }
    return levels_.front().space->continuity();
}

int AdaptiveSpace::value_dimension() const noexcept {
    if (levels_.empty() || !levels_.front().space) {
        return 0;
    }
    return levels_.front().space->value_dimension();
}

int AdaptiveSpace::topological_dimension() const noexcept {
    if (levels_.empty() || !levels_.front().space) {
        return -1;
    }
    return levels_.front().space->topological_dimension();
}

int AdaptiveSpace::polynomial_order() const noexcept {
    if (levels_.empty() || !levels_.front().space) {
        return 0;
    }
    return active_level().order;
}

ElementType AdaptiveSpace::element_type() const noexcept {
    if (levels_.empty() || !levels_.front().space) {
        return ElementType::Unknown;
    }
    return levels_.front().space->element_type();
}

const elements::Element& AdaptiveSpace::element() const noexcept {
    return active_level().space->element();
}

std::shared_ptr<const elements::Element> AdaptiveSpace::element_ptr() const noexcept {
    if (levels_.empty() || !levels_.front().space) {
        return {};
    }
    return active_level().space->element_ptr();
}

std::size_t AdaptiveSpace::dofs_per_element() const noexcept {
    if (levels_.empty() || !levels_.front().space) {
        return 0;
    }
    return active_level().space->dofs_per_element();
}

FunctionSpace::Value AdaptiveSpace::evaluate(const Value& xi,
                                             const std::vector<Real>& coefficients) const {
    return active_level().space->evaluate(xi, coefficients);
}

void AdaptiveSpace::interpolate(const ValueFunction& function,
                                std::vector<Real>& coefficients) const {
    active_level().space->interpolate(function, coefficients);
}

} // namespace spaces
} // namespace FE
} // namespace svmp

