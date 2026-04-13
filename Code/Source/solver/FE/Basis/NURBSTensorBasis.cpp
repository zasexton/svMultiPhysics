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
                                            std::vector<Gradient>* gradients,
                                            std::vector<Hessian>* hessians) const {
    values.assign(size_, Real(0));
    if (gradients != nullptr) {
        gradients->assign(size_, Gradient{});
    }
    if (hessians != nullptr) {
        hessians->assign(size_, Hessian{});
    }

    std::vector<std::vector<Real>> axis_values(static_cast<std::size_t>(dimension_));
    std::vector<std::vector<Gradient>> axis_gradients(static_cast<std::size_t>(dimension_));
    std::vector<std::vector<Hessian>> axis_hessians(static_cast<std::size_t>(dimension_));
    for (int axis = 0; axis < dimension_; ++axis) {
        axes_[static_cast<std::size_t>(axis)].evaluate_values(axis_coordinate(xi, axis),
                                                              axis_values[static_cast<std::size_t>(axis)]);
        if (gradients != nullptr || hessians != nullptr) {
            axes_[static_cast<std::size_t>(axis)].evaluate_gradients(axis_coordinate(xi, axis),
                                                                     axis_gradients[static_cast<std::size_t>(axis)]);
        }
        if (hessians != nullptr) {
            axes_[static_cast<std::size_t>(axis)].evaluate_hessians(axis_coordinate(xi, axis),
                                                                    axis_hessians[static_cast<std::size_t>(axis)]);
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
                if (hessians != nullptr) {
                    (*hessians)[idx](0, 0) = axis_hessians[0][i](0, 0) * axis_values[1][j];
                    (*hessians)[idx](1, 1) = axis_values[0][i] * axis_hessians[1][j](0, 0);
                    const Real cross = axis_gradients[0][i][0] * axis_gradients[1][j][0];
                    (*hessians)[idx](0, 1) = cross;
                    (*hessians)[idx](1, 0) = cross;
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
                if (hessians != nullptr) {
                    (*hessians)[idx](0, 0) =
                        axis_hessians[0][i](0, 0) * axis_values[1][j] * axis_values[2][k];
                    (*hessians)[idx](1, 1) =
                        axis_values[0][i] * axis_hessians[1][j](0, 0) * axis_values[2][k];
                    (*hessians)[idx](2, 2) =
                        axis_values[0][i] * axis_values[1][j] * axis_hessians[2][k](0, 0);
                    const Real dxy =
                        axis_gradients[0][i][0] * axis_gradients[1][j][0] * axis_values[2][k];
                    const Real dxz =
                        axis_gradients[0][i][0] * axis_values[1][j] * axis_gradients[2][k][0];
                    const Real dyz =
                        axis_values[0][i] * axis_gradients[1][j][0] * axis_gradients[2][k][0];
                    (*hessians)[idx](0, 1) = dxy;
                    (*hessians)[idx](1, 0) = dxy;
                    (*hessians)[idx](0, 2) = dxz;
                    (*hessians)[idx](2, 0) = dxz;
                    (*hessians)[idx](1, 2) = dyz;
                    (*hessians)[idx](2, 1) = dyz;
                }
            }
        }
    }
}

void NURBSTensorBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                       std::vector<Real>& values) const {
    std::vector<Real> nonrational;
    evaluate_nonrational(xi, nonrational, nullptr, nullptr);

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
    evaluate_nonrational(xi, nonrational, &nonrational_gradients, nullptr);

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

void NURBSTensorBasis::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                         std::vector<Hessian>& hessians) const {
    std::vector<Real> nonrational;
    std::vector<Gradient> nonrational_gradients;
    std::vector<Hessian> nonrational_hessians;
    evaluate_nonrational(xi, nonrational, &nonrational_gradients, &nonrational_hessians);

    Real denom = Real(0);
    Gradient denom_gradient{};
    Hessian denom_hessian{};
    for (std::size_t i = 0; i < size_; ++i) {
        const Real weight = weights_[i];
        denom += nonrational[i] * weight;
        for (int a = 0; a < dimension_; ++a) {
            const std::size_t sa = static_cast<std::size_t>(a);
            denom_gradient[sa] += nonrational_gradients[i][sa] * weight;
            for (int b = 0; b < dimension_; ++b) {
                const std::size_t sb = static_cast<std::size_t>(b);
                denom_hessian(sa, sb) += nonrational_hessians[i](sa, sb) * weight;
            }
        }
    }

    const Real eps = std::numeric_limits<Real>::epsilon() * Real(64);
    if (std::abs(denom) <= eps) {
        throw BasisEvaluationException("NURBSTensorBasis: rational denominator is zero",
                                       __FILE__, __LINE__, __func__);
    }

    hessians.assign(size_, Hessian{});
    const Real inv_denom = Real(1) / denom;
    const Real inv_denom_sq = inv_denom * inv_denom;
    const Real inv_denom_cu = inv_denom_sq * inv_denom;
    for (std::size_t i = 0; i < size_; ++i) {
        const Real weight = weights_[i];
        const Real A = nonrational[i] * weight;
        for (int a = 0; a < dimension_; ++a) {
            const std::size_t sa = static_cast<std::size_t>(a);
            const Real dA_a = nonrational_gradients[i][sa] * weight;
            for (int b = 0; b < dimension_; ++b) {
                const std::size_t sb = static_cast<std::size_t>(b);
                const Real dA_b = nonrational_gradients[i][sb] * weight;
                const Real ddA = nonrational_hessians[i](sa, sb) * weight;
                hessians[i](sa, sb) =
                    (ddA * denom * denom
                     - A * denom_hessian(sa, sb) * denom
                     - dA_a * denom * denom_gradient[sb]
                     - dA_b * denom * denom_gradient[sa]
                     + Real(2) * A * denom_gradient[sa] * denom_gradient[sb]) *
                    inv_denom_cu;
            }
        }
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
