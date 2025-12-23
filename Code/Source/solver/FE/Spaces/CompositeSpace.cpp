/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/CompositeSpace.h"

namespace svmp {
namespace FE {
namespace spaces {

void CompositeSpace::add_region(int region_id,
                                std::shared_ptr<FunctionSpace> space) {
    FE_CHECK_NOT_NULL(space.get(), "CompositeSpace::add_region space");
    // Ensure uniqueness of region_id
    for (const auto& r : regions_) {
        if (r.region_id == region_id) {
            FE_THROW(InvalidArgumentException,
                     "CompositeSpace::add_region: duplicate region_id");
        }
    }
    regions_.push_back(RegionSpace{region_id, std::move(space)});
}

const FunctionSpace& CompositeSpace::space_for_region(int region_id) const {
    for (const auto& r : regions_) {
        if (r.region_id == region_id && r.space) {
            return *r.space;
        }
    }
    FE_THROW(InvalidArgumentException,
             "CompositeSpace::space_for_region: unknown region_id");
}

std::shared_ptr<const FunctionSpace> CompositeSpace::try_space_for_region(int region_id) const {
    for (const auto& r : regions_) {
        if (r.region_id == region_id) {
            return r.space;
        }
    }
    return {};
}

FieldType CompositeSpace::field_type() const noexcept {
    if (regions_.empty() || !regions_.front().space) {
        return FieldType::Mixed;
    }
    return regions_.front().space->field_type();
}

Continuity CompositeSpace::continuity() const noexcept {
    if (regions_.empty() || !regions_.front().space) {
        return Continuity::Custom;
    }
    return regions_.front().space->continuity();
}

int CompositeSpace::value_dimension() const noexcept {
    if (regions_.empty() || !regions_.front().space) {
        return 0;
    }
    return regions_.front().space->value_dimension();
}

int CompositeSpace::topological_dimension() const noexcept {
    if (regions_.empty() || !regions_.front().space) {
        return -1;
    }
    return regions_.front().space->topological_dimension();
}

int CompositeSpace::polynomial_order() const noexcept {
    int max_order = 0;
    for (const auto& r : regions_) {
        if (r.space) {
            max_order = std::max(max_order, r.space->polynomial_order());
        }
    }
    return max_order;
}

ElementType CompositeSpace::element_type() const noexcept {
    if (regions_.empty() || !regions_.front().space) {
        return ElementType::Unknown;
    }
    return regions_.front().space->element_type();
}

const elements::Element& CompositeSpace::element() const noexcept {
    return regions_.front().space->element();
}

std::shared_ptr<const elements::Element> CompositeSpace::element_ptr() const noexcept {
    if (regions_.empty() || !regions_.front().space) {
        return {};
    }
    return regions_.front().space->element_ptr();
}

} // namespace spaces
} // namespace FE
} // namespace svmp

