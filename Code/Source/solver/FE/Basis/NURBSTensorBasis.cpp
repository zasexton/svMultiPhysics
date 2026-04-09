/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Basis/NURBSTensorBasis.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace svmp {
namespace FE {
namespace basis {

namespace {

math::Vector<Real, 3> axis_coordinate(const math::Vector<Real, 3>& xi, int axis) {
    math::Vector<Real, 3> coord{};
    coord[0] = xi[static_cast<std::size_t>(axis)];
    return coord;
}

} // namespace

NURBSTensorBasis::NURBSTensorBasis(BSplineBasis bx,
                                   BSplineBasis by,
                                   std::vector<Real> weights,
                                   std::vector<int> tensor_extents) {
    initialize({std::move(bx), std::move(by)}, std::move(weights), std::move(tensor_extents));
}

NURBSTensorBasis::NURBSTensorBasis(BSplineBasis bx,
                                   BSplineBasis by,
                                   BSplineBasis bz,
                                   std::vector<Real> weights,
                                   std::vector<int> tensor_extents) {
    initialize({std::move(bx), std::move(by), std::move(bz)}, std::move(weights), std::move(tensor_extents));
}

void NURBSTensorBasis::initialize(std::vector<BSplineBasis> axes,
                                  std::vector<Real> weights,
                                  std::vector<int> tensor_extents) {
    if (axes.size() != 2u && axes.size() != 3u) {
        throw BasisConfigurationException("NURBSTensorBasis requires 2D or 3D tensor-product axes",
                                          __FILE__, __LINE__, __func__);
    }

    dimension_ = static_cast<int>(axes.size());
    element_type_ = (dimension_ == 2) ? ElementType::Quad4 : ElementType::Hex8;
    order_ = 0;
    size_ = 1u;

    axes_ = std::move(axes);
    axis_sizes_.resize(static_cast<std::size_t>(dimension_));
    tensor_extents_.resize(static_cast<std::size_t>(dimension_));

    for (int axis = 0; axis < dimension_; ++axis) {
        if (axes_[static_cast<std::size_t>(axis)].is_rational()) {
            throw BasisConfigurationException("NURBSTensorBasis expects non-rational BSpline axes and a separate control-net weight array",
                                              __FILE__, __LINE__, __func__);
        }

        const std::size_t axis_size = axes_[static_cast<std::size_t>(axis)].size();
        axis_sizes_[static_cast<std::size_t>(axis)] = axis_size;
        order_ = std::max(order_, axes_[static_cast<std::size_t>(axis)].order());
        size_ *= axis_size;
    }

    if (weights.size() != size_) {
        throw BasisConfigurationException("NURBSTensorBasis: weights size must match tensor-product basis size",
                                          __FILE__, __LINE__, __func__);
    }

    if (!tensor_extents.empty()) {
        if (tensor_extents.size() != static_cast<std::size_t>(dimension_)) {
            throw BasisConfigurationException("NURBSTensorBasis: tensor_extents must match tensor dimension",
                                              __FILE__, __LINE__, __func__);
        }
        for (int axis = 0; axis < dimension_; ++axis) {
            if (tensor_extents[static_cast<std::size_t>(axis)] != static_cast<int>(axis_sizes_[static_cast<std::size_t>(axis)])) {
                throw BasisConfigurationException("NURBSTensorBasis: tensor_extents must agree with spline axis basis sizes",
                                                  __FILE__, __LINE__, __func__);
            }
        }
        tensor_extents_ = std::move(tensor_extents);
    } else {
        for (int axis = 0; axis < dimension_; ++axis) {
            tensor_extents_[static_cast<std::size_t>(axis)] =
                static_cast<int>(axis_sizes_[static_cast<std::size_t>(axis)]);
        }
    }

    weights_ = std::move(weights);
}

std::string NURBSTensorBasis::cache_identity() const {
    std::ostringstream oss;
    oss << BasisFunction::cache_identity()
        << "|axes=" << dimension_;
    for (int axis = 0; axis < dimension_; ++axis) {
        oss << "|axis" << axis << '=' << axes_[static_cast<std::size_t>(axis)].cache_identity()
            << "|extent" << axis << '=' << tensor_extents_[static_cast<std::size_t>(axis)];
    }

    oss << std::setprecision(std::numeric_limits<Real>::max_digits10);
    for (Real weight : weights_) {
        oss << "|w=" << weight;
    }
    return oss.str();
}

void NURBSTensorBasis::evaluate_nonrational(const math::Vector<Real, 3>& xi,
                                            std::vector<Real>& values,
                                            std::vector<Gradient>* gradients) const {
    values.assign(size_, Real(0));
    if (gradients != nullptr) {
        gradients->assign(size_, Gradient{});
    }

    std::vector<std::vector<Real>> axis_values(static_cast<std::size_t>(dimension_));
    std::vector<std::vector<Gradient>> axis_gradients(static_cast<std::size_t>(dimension_));
    for (int axis = 0; axis < dimension_; ++axis) {
        axes_[static_cast<std::size_t>(axis)].evaluate_values(axis_coordinate(xi, axis),
                                                              axis_values[static_cast<std::size_t>(axis)]);
        if (gradients != nullptr) {
            axes_[static_cast<std::size_t>(axis)].evaluate_gradients(axis_coordinate(xi, axis),
                                                                     axis_gradients[static_cast<std::size_t>(axis)]);
        }
    }

    if (dimension_ == 2) {
        const std::size_t nx = axis_sizes_[0];
        const std::size_t ny = axis_sizes_[1];
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const std::size_t idx = j * nx + i;
                values[idx] = axis_values[0][i] * axis_values[1][j];
                if (gradients != nullptr) {
                    (*gradients)[idx][0] = axis_gradients[0][i][0] * axis_values[1][j];
                    (*gradients)[idx][1] = axis_values[0][i] * axis_gradients[1][j][0];
                }
            }
        }
        return;
    }

    const std::size_t nx = axis_sizes_[0];
    const std::size_t ny = axis_sizes_[1];
    const std::size_t nz = axis_sizes_[2];
    for (std::size_t k = 0; k < nz; ++k) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const std::size_t idx = (k * ny + j) * nx + i;
                values[idx] = axis_values[0][i] * axis_values[1][j] * axis_values[2][k];
                if (gradients != nullptr) {
                    (*gradients)[idx][0] =
                        axis_gradients[0][i][0] * axis_values[1][j] * axis_values[2][k];
                    (*gradients)[idx][1] =
                        axis_values[0][i] * axis_gradients[1][j][0] * axis_values[2][k];
                    (*gradients)[idx][2] =
                        axis_values[0][i] * axis_values[1][j] * axis_gradients[2][k][0];
                }
            }
        }
    }
}

void NURBSTensorBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                       std::vector<Real>& values) const {
    std::vector<Real> nonrational;
    evaluate_nonrational(xi, nonrational, nullptr);

    Real denom = Real(0);
    values.assign(size_, Real(0));
    for (std::size_t i = 0; i < size_; ++i) {
        values[i] = nonrational[i] * weights_[i];
        denom += values[i];
    }

    const Real eps = std::numeric_limits<Real>::epsilon() * Real(64);
    if (std::abs(denom) <= eps) {
        throw BasisEvaluationException("NURBSTensorBasis: rational denominator is zero",
                                       __FILE__, __LINE__, __func__);
    }

    const Real inv_denom = Real(1) / denom;
    for (Real& value : values) {
        value *= inv_denom;
    }
}

void NURBSTensorBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                          std::vector<Gradient>& gradients) const {
    std::vector<Real> nonrational;
    std::vector<Gradient> nonrational_gradients;
    evaluate_nonrational(xi, nonrational, &nonrational_gradients);

    Real denom = Real(0);
    Gradient denom_gradient{};
    for (std::size_t i = 0; i < size_; ++i) {
        denom += nonrational[i] * weights_[i];
        for (int d = 0; d < dimension_; ++d) {
            denom_gradient[static_cast<std::size_t>(d)] +=
                nonrational_gradients[i][static_cast<std::size_t>(d)] * weights_[i];
        }
    }

    const Real eps = std::numeric_limits<Real>::epsilon() * Real(64);
    if (std::abs(denom) <= eps) {
        throw BasisEvaluationException("NURBSTensorBasis: rational denominator is zero",
                                       __FILE__, __LINE__, __func__);
    }

    gradients.assign(size_, Gradient{});
    const Real inv_denom_sq = Real(1) / (denom * denom);
    for (std::size_t i = 0; i < size_; ++i) {
        const Real weighted_value = nonrational[i] * weights_[i];
        for (int d = 0; d < dimension_; ++d) {
            const std::size_t sd = static_cast<std::size_t>(d);
            const Real weighted_grad = nonrational_gradients[i][sd] * weights_[i];
            gradients[i][sd] =
                (weighted_grad * denom - weighted_value * denom_gradient[sd]) * inv_denom_sq;
        }
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
