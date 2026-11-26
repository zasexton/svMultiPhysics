/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_GEOMETRY_SUBPARAMETRICMAPPING_H
#define SVMP_FE_GEOMETRY_SUBPARAMETRICMAPPING_H

/**
 * @file SubparametricMapping.h
 * @brief Geometry mapping with reduced geometric order (wrapper around isoparametric mapping)
 */

#include "IsoparametricMapping.h"

namespace svmp {
namespace FE {
namespace geometry {

class SubparametricMapping : public IsoparametricMapping {
public:
    SubparametricMapping(std::shared_ptr<basis::BasisFunction> geometry_basis,
                         std::vector<math::Vector<Real, 3>> nodes,
                         int geometry_order,
                         std::size_t parent_node_count = 0)
        : IsoparametricMapping(std::move(geometry_basis), std::move(nodes)), geometry_order_(geometry_order),
          parent_node_count_(parent_node_count) {}

    int geometry_order() const noexcept { return geometry_order_; }

    void resetNodes(const std::vector<math::Vector<Real, 3>>& nodes) override {
        if (nodes.size() == num_nodes()) {
            IsoparametricMapping::resetNodes(nodes);
            return;
        }
        FE_THROW_IF(parent_node_count_ == 0 || nodes.size() != parent_node_count_, FEException,
                    "SubparametricMapping: node count does not match vertex or parent geometry");
        IsoparametricMapping::resetNodes({nodes.begin(), nodes.begin() + num_nodes()});
    }

private:
    int geometry_order_;
    std::size_t parent_node_count_;
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_SUBPARAMETRICMAPPING_H
