/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "IsoparametricMapping.h"
#include "InverseMapping.h"
#include "Core/FEException.h"
#include "GeometryFrameUtils.h"
#include <numeric>
#include <cmath>

namespace svmp {
namespace FE {
namespace geometry {

namespace {

constexpr Real kDegenerateTol = detail::kDegenerateTol;

} // namespace

IsoparametricMapping::IsoparametricMapping(std::shared_ptr<basis::BasisFunction> basis,
                                           std::vector<math::Vector<Real, 3>> nodes)
    : basis_(std::move(basis)), nodes_(std::move(nodes)) {
    if (!basis_) {
        throw FEException("IsoparametricMapping requires a basis",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (basis_->size() != nodes_.size()) {
        throw FEException("IsoparametricMapping node count does not match basis size",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

math::Vector<Real, 3> IsoparametricMapping::map_to_physical(const math::Vector<Real, 3>& xi) const {
    std::vector<Real> N;
    basis_->evaluate_values(xi, N);
    math::Vector<Real, 3> x{};
    for (std::size_t a = 0; a < N.size(); ++a) {
        x += nodes_[a] * N[a];
    }
    return x;
}

math::Matrix<Real, 3, 3> IsoparametricMapping::jacobian(const math::Vector<Real, 3>& xi) const {
    std::vector<basis::Gradient> grads;
    basis_->evaluate_gradients(xi, grads);
    math::Matrix<Real, 3, 3> J{};
    const int dim = basis_->dimension();
    for (std::size_t a = 0; a < grads.size(); ++a) {
        for (int j = 0; j < dim; ++j) {
            const std::size_t sj = static_cast<std::size_t>(j);
            for (std::size_t i = 0; i < 3; ++i) {
                J(i, sj) += nodes_[a][i] * grads[a][sj];
            }
        }
    }

    // For dim < 3 (curves/surfaces), complete a full 3x3 frame Jacobian:
    // columns 0..dim-1 are true tangents, remaining columns are an orthonormal
    // complement to make the matrix invertible.
    if (dim == 1) {
        const math::Vector<Real, 3> t{J(0, 0), J(1, 0), J(2, 0)};
        math::Vector<Real, 3> n1{};
        math::Vector<Real, 3> n2{};
        detail::complete_curve_frame(t, n1, n2);
        J(0, 1) = n1[0]; J(1, 1) = n1[1]; J(2, 1) = n1[2];
        J(0, 2) = n2[0]; J(1, 2) = n2[1]; J(2, 2) = n2[2];
    } else if (dim == 2) {
        const math::Vector<Real, 3> tu{J(0, 0), J(1, 0), J(2, 0)};
        const math::Vector<Real, 3> tv{J(0, 1), J(1, 1), J(2, 1)};
        const auto n = tu.cross(tv);
        const Real n_norm = n.norm();
        if (n_norm < kDegenerateTol) {
            // Degenerate surface: keep third column zero so det(J)=0.
            J(0, 2) = Real(0); J(1, 2) = Real(0); J(2, 2) = Real(0);
        } else {
            const auto n_unit = n / n_norm;
            J(0, 2) = n_unit[0]; J(1, 2) = n_unit[1]; J(2, 2) = n_unit[2];
        }
    }
    return J;
}

math::Vector<Real, 3> IsoparametricMapping::map_to_reference(const math::Vector<Real, 3>& x_phys,
                                                            const math::Vector<Real, 3>& initial_guess) const {
    return InverseMapping::solve(*this, x_phys, initial_guess);
}

} // namespace geometry
} // namespace FE
} // namespace svmp
