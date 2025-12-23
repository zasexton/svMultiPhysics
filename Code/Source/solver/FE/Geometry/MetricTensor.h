/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_GEOMETRY_METRICTENSOR_H
#define SVMP_FE_GEOMETRY_METRICTENSOR_H

/**
 * @file MetricTensor.h
 * @brief Metric tensor utilities derived from Jacobians
 */

#include "Core/Types.h"
#include "Math/Matrix.h"

namespace svmp {
namespace FE {
namespace geometry {

class MetricTensor {
public:
    /// Compute covariant metric G = J^T J (truncated to dimension)
    static math::Matrix<Real, 3, 3> covariant(const math::Matrix<Real, 3, 3>& J, int dim);

    /// Compute contravariant metric (inverse of covariant)
    static math::Matrix<Real, 3, 3> contravariant(const math::Matrix<Real, 3, 3>& J, int dim);
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_METRICTENSOR_H
