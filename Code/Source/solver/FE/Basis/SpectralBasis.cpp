/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "SpectralBasis.h"
#include "Quadrature/GaussLobattoQuadrature.h"
#include <cmath>

namespace svmp {
namespace FE {
namespace basis {

namespace {

bool is_line(ElementType t) {
    return t == ElementType::Line2 || t == ElementType::Line3;
}

bool is_quadrilateral(ElementType t) {
    return t == ElementType::Quad4 || t == ElementType::Quad8 || t == ElementType::Quad9;
}

bool is_hexahedron(ElementType t) {
    return t == ElementType::Hex8 || t == ElementType::Hex20 || t == ElementType::Hex27;
}

bool is_wedge(ElementType t) {
    return t == ElementType::Wedge6 || t == ElementType::Wedge15 || t == ElementType::Wedge18;
}

bool is_pyramid(ElementType t) {
    return t == ElementType::Pyramid5 || t == ElementType::Pyramid13 || t == ElementType::Pyramid14;
}

} // namespace

SpectralBasis::SpectralBasis(ElementType type, int order)
    : element_type_(type), dimension_(0), order_(order), size_(0) {
    if (order_ < 1) {
        order_ = 1; // spectral elements require at least quadratic to expose GLL structure
    }

    if (is_line(element_type_)) {
        dimension_ = 1;
        size_ = static_cast<std::size_t>(order_ + 1);
        build_nodes();
    } else if (is_quadrilateral(element_type_)) {
        dimension_ = 2;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 1));
        build_nodes();
    } else if (is_hexahedron(element_type_)) {
        dimension_ = 3;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 1) * (order_ + 1));
        build_nodes();
    } else if (is_wedge(element_type_) || is_pyramid(element_type_)) {
        throw FEException("SpectralBasis currently supports only tensor-product elements (line/quad/hex)",
                          __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
    } else {
        throw FEException("SpectralBasis supports only line/quad/hex reference elements",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

void SpectralBasis::build_nodes() {
    auto raw = quadrature::GaussLobattoQuadrature1D::generate_raw(order_ + 1);
    nodes_1d_ = raw.first;
    barycentric_weights_.assign(nodes_1d_.size(), Real(1));
    for (std::size_t i = 0; i < nodes_1d_.size(); ++i) {
        Real prod = Real(1);
        for (std::size_t j = 0; j < nodes_1d_.size(); ++j) {
            if (i == j) continue;
            prod *= nodes_1d_[i] - nodes_1d_[j];
        }
        barycentric_weights_[i] = Real(1) / prod;
    }
}

std::vector<Real> SpectralBasis::eval_1d(Real x) const {
    std::vector<Real> vals(nodes_1d_.size(), Real(0));
    const std::size_t n = nodes_1d_.size();
    for (std::size_t i = 0; i < n; ++i) {
        Real v = Real(1);
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            v *= (x - nodes_1d_[j]) / (nodes_1d_[i] - nodes_1d_[j]);
        }
        vals[i] = v;
    }
    return vals;
}

std::vector<Real> SpectralBasis::eval_1d_derivative(Real x) const {
    std::vector<Real> ders(nodes_1d_.size(), Real(0));
    const std::size_t n = nodes_1d_.size();
    for (std::size_t i = 0; i < n; ++i) {
        Real sum = Real(0);
        for (std::size_t m = 0; m < n; ++m) {
            if (m == i) continue;
            Real prod = Real(1);
            for (std::size_t j = 0; j < n; ++j) {
                if (j == i || j == m) continue;
                prod *= (x - nodes_1d_[j]) / (nodes_1d_[i] - nodes_1d_[j]);
            }
            prod /= (nodes_1d_[i] - nodes_1d_[m]);
            sum += prod;
        }
        ders[i] = sum;
    }
    return ders;
}

void SpectralBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                    std::vector<Real>& values) const {
    values.assign(size_, Real(0));

    if (dimension_ == 1) {
        values = eval_1d(xi[0]);
        return;
    }

    if (dimension_ == 2) {
        auto lx = eval_1d(xi[0]);
        auto ly = eval_1d(xi[1]);
        std::size_t idx = 0;
        for (std::size_t j = 0; j < ly.size(); ++j) {
            for (std::size_t i = 0; i < lx.size(); ++i) {
                values[idx++] = lx[i] * ly[j];
            }
        }
        return;
    }

    auto lx = eval_1d(xi[0]);
    auto ly = eval_1d(xi[1]);
    auto lz = eval_1d(xi[2]);
    std::size_t idx = 0;
    for (std::size_t k = 0; k < lz.size(); ++k) {
        for (std::size_t j = 0; j < ly.size(); ++j) {
            for (std::size_t i = 0; i < lx.size(); ++i) {
                values[idx++] = lx[i] * ly[j] * lz[k];
            }
        }
    }
}

void SpectralBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                       std::vector<Gradient>& gradients) const {
    gradients.assign(size_, Gradient{});

    if (dimension_ == 1) {
        auto d = eval_1d_derivative(xi[0]);
        for (std::size_t i = 0; i < d.size(); ++i) {
            gradients[i][0] = d[i];
        }
        return;
    }

    if (dimension_ == 2) {
        auto lx = eval_1d(xi[0]);
        auto ly = eval_1d(xi[1]);
        auto dx = eval_1d_derivative(xi[0]);
        auto dy = eval_1d_derivative(xi[1]);
        std::size_t idx = 0;
        for (std::size_t j = 0; j < ly.size(); ++j) {
            for (std::size_t i = 0; i < lx.size(); ++i) {
                gradients[idx][0] = dx[i] * ly[j];
                gradients[idx][1] = lx[i] * dy[j];
                ++idx;
            }
        }
        return;
    }

    auto lx = eval_1d(xi[0]);
    auto ly = eval_1d(xi[1]);
    auto lz = eval_1d(xi[2]);
    auto dx = eval_1d_derivative(xi[0]);
    auto dy = eval_1d_derivative(xi[1]);
    auto dz = eval_1d_derivative(xi[2]);

    std::size_t idx = 0;
    for (std::size_t k = 0; k < lz.size(); ++k) {
        for (std::size_t j = 0; j < ly.size(); ++j) {
            for (std::size_t i = 0; i < lx.size(); ++i) {
                gradients[idx][0] = dx[i] * ly[j] * lz[k];
                gradients[idx][1] = lx[i] * dy[j] * lz[k];
                gradients[idx][2] = lx[i] * ly[j] * dz[k];
                ++idx;
            }
        }
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
