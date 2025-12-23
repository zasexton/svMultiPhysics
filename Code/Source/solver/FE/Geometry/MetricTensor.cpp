/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "MetricTensor.h"

namespace svmp {
namespace FE {
namespace geometry {

math::Matrix<Real, 3, 3> MetricTensor::covariant(const math::Matrix<Real, 3, 3>& J, int dim) {
    math::Matrix<Real, 3, 3> G{};
    const std::size_t sdim = static_cast<std::size_t>(dim);
    for (std::size_t i = 0; i < sdim; ++i) {
        for (std::size_t j = 0; j < sdim; ++j) {
            Real val = Real(0);
            for (std::size_t k = 0; k < 3; ++k) {
                val += J(k, i) * J(k, j);
            }
            G(i, j) = val;
        }
    }
    return G;
}

math::Matrix<Real, 3, 3> MetricTensor::contravariant(const math::Matrix<Real, 3, 3>& J, int dim) {
    auto G = covariant(J, dim);
    if (dim == 1) {
        math::Matrix<Real, 3, 3> inv{};
        inv(0,0) = Real(1) / G(0,0);
        return inv;
    }
    if (dim == 2) {
        math::Matrix<Real, 2, 2> sub{};
        sub(0,0) = G(0,0); sub(0,1) = G(0,1);
        sub(1,0) = G(1,0); sub(1,1) = G(1,1);
        auto sub_inv = sub.inverse();
        math::Matrix<Real, 3, 3> inv{};
        inv(0,0) = sub_inv(0,0); inv(0,1) = sub_inv(0,1);
        inv(1,0) = sub_inv(1,0); inv(1,1) = sub_inv(1,1);
        return inv;
    }
    return G.inverse();
}

} // namespace geometry
} // namespace FE
} // namespace svmp
