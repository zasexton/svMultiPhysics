/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_GEOMETRY_GEOMETRYVALIDATOR_H
#define SVMP_FE_GEOMETRY_GEOMETRYVALIDATOR_H

/**
 * @file GeometryValidator.h
 * @brief Light-weight mapping quality checks
 */

#include "GeometryMapping.h"

namespace svmp {
namespace FE {
namespace geometry {

struct GeometryQuality {
    Real detJ;
    Real condition_number;
    bool positive_jacobian;
};

class GeometryValidator {
public:
    static GeometryQuality evaluate(const GeometryMapping& mapping,
                                    const math::Vector<Real, 3>& xi);
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_GEOMETRYVALIDATOR_H
