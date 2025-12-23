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
                         int geometry_order)
        : IsoparametricMapping(std::move(geometry_basis), std::move(nodes)), geometry_order_(geometry_order) {}

    int geometry_order() const noexcept { return geometry_order_; }

private:
    int geometry_order_;
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_SUBPARAMETRICMAPPING_H
