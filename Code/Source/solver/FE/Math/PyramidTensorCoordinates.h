/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_MATH_PYRAMIDTENSORCOORDINATES_H
#define SVMP_FE_MATH_PYRAMIDTENSORCOORDINATES_H

#include "Core/Types.h"
#include "Math/Matrix.h"
#include "Math/Vector.h"

#include <array>
#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace math {

struct PyramidTensorCoordinate {
    Real value{Real(0)};
    Vector<Real, 3> first{};
    Matrix<Real, 3, 3> second{};
};

inline bool pyramid_tensor_near_apex(Real one_minus_z) noexcept {
    const Real scale = std::max(Real(1), std::abs(one_minus_z));
    return std::abs(one_minus_z) <=
           Real(64) * std::numeric_limits<Real>::epsilon() * scale;
}

inline std::array<PyramidTensorCoordinate, 3> pyramid_tensor_coordinates(
    const Vector<Real, 3>& xi) {
    std::array<PyramidTensorCoordinate, 3> coordinates{};

    auto& tx = coordinates[0];
    auto& ty = coordinates[1];
    auto& tz = coordinates[2];

    const Real z = xi[2];
    const Real one_minus_z = Real(1) - z;
    tx.value = Real(0.5);
    ty.value = Real(0.5);
    if (!pyramid_tensor_near_apex(one_minus_z)) {
        const Real inv = Real(1) / one_minus_z;
        const Real inv2 = inv * inv;
        const Real inv3 = inv2 * inv;

        tx.value = Real(0.5) * (xi[0] * inv + Real(1));
        tx.first[0] = Real(0.5) * inv;
        tx.first[2] = Real(0.5) * xi[0] * inv2;
        tx.second(0u, 2u) = Real(0.5) * inv2;
        tx.second(2u, 0u) = tx.second(0u, 2u);
        tx.second(2u, 2u) = xi[0] * inv3;

        ty.value = Real(0.5) * (xi[1] * inv + Real(1));
        ty.first[1] = Real(0.5) * inv;
        ty.first[2] = Real(0.5) * xi[1] * inv2;
        ty.second(1u, 2u) = Real(0.5) * inv2;
        ty.second(2u, 1u) = ty.second(1u, 2u);
        ty.second(2u, 2u) = xi[1] * inv3;
    }

    tz.value = Real(0.5) * (z + Real(1));
    tz.first[2] = Real(0.5);
    return coordinates;
}

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_PYRAMIDTENSORCOORDINATES_H
