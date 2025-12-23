/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_COMPOSITESPACE_H
#define SVMP_FE_SPACES_COMPOSITESPACE_H

/**
 * @file CompositeSpace.h
 * @brief Composite spaces with region-dependent function spaces
 *
 * CompositeSpace associates different FunctionSpace instances with abstract
 * region identifiers (e.g., subdomains or material regions). It is a pure
 * FE-layer construct and does not know about Mesh topology; higher-level
 * modules map mesh region markers to the region ids used here.
 */

#include "Spaces/FunctionSpace.h"
#include <vector>

namespace svmp {
namespace FE {
namespace spaces {

class CompositeSpace : public FunctionSpace {
public:
    struct RegionSpace {
        int region_id;
        std::shared_ptr<FunctionSpace> space;
    };

    CompositeSpace() = default;

    /// Register a function space for a given region id
    void add_region(int region_id,
                    std::shared_ptr<FunctionSpace> space);

    /// Number of registered regions
    std::size_t num_regions() const noexcept { return regions_.size(); }

    /// Access region space descriptor
    const RegionSpace& region(std::size_t i) const { return regions_.at(i); }

    /// Get space for region id (throws if not found)
    const FunctionSpace& space_for_region(int region_id) const;

    /// Optional access returning nullptr if not found
    std::shared_ptr<const FunctionSpace> try_space_for_region(int region_id) const;

    // FunctionSpace interface, forwarded to the first registered region
    SpaceType space_type() const noexcept override { return SpaceType::Composite; }
    FieldType field_type() const noexcept override;
    Continuity continuity() const noexcept override;

    int value_dimension() const noexcept override;
    int topological_dimension() const noexcept override;
    int polynomial_order() const noexcept override;
    ElementType element_type() const noexcept override;

    const elements::Element& element() const noexcept override;
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override;

    /// Composite spaces do not implement a unified evaluation
    Value evaluate(const Value&,
                   const std::vector<Real>&) const override {
        FE_THROW(NotImplementedException,
                 "CompositeSpace::evaluate is not defined for aggregated regions");
    }

private:
    std::vector<RegionSpace> regions_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_COMPOSITESPACE_H

