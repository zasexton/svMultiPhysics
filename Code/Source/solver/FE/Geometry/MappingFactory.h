/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_GEOMETRY_MAPPINGFACTORY_H
#define SVMP_FE_GEOMETRY_MAPPINGFACTORY_H

/**
 * @file MappingFactory.h
 * @brief Factory for geometry mappings
 */

#include "GeometryMapping.h"
#include "LinearMapping.h"
#include "IsoparametricMapping.h"
#include "SubparametricMapping.h"
#include "SuperparametricMapping.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/SerendipityBasis.h"
#include "Core/FEException.h"
#include <memory>

namespace svmp {
namespace FE {
namespace geometry {

struct MappingRequest {
    ElementType element_type;
    int geometry_order{1};
    bool use_affine{true};
    std::shared_ptr<basis::BasisFunction> basis; // optional
};

class MappingFactory {
public:
    static std::shared_ptr<GeometryMapping> create(const MappingRequest& req,
                                                   const std::vector<math::Vector<Real, 3>>& nodes);
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_MAPPINGFACTORY_H
