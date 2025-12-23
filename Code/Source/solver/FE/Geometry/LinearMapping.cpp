/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "LinearMapping.h"
#include "InverseMapping.h"
#include "GeometryFrameUtils.h"
#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace geometry {

namespace {

bool is_simplex(ElementType type) {
    return type == ElementType::Line2 || type == ElementType::Triangle3 || type == ElementType::Tetra4;
}

constexpr Real kDegenerateTol = detail::kDegenerateTol;

}

LinearMapping::LinearMapping(ElementType type,
                             std::vector<math::Vector<Real, 3>> nodes)
    : element_type_(type), dimension_(element_dimension(type)), nodes_(std::move(nodes)) {
    if (!is_simplex(type) && type != ElementType::Line2) {
        throw FEException("LinearMapping supports affine simplex elements (Line2, Triangle3, Tetra4)",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }
    if (nodes_.empty()) {
        throw FEException("LinearMapping requires nodal coordinates",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    basis_ = std::make_shared<basis::LagrangeBasis>(type, 1);
    if (basis_->size() != nodes_.size()) {
        throw FEException("LinearMapping node count does not match basis size",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

math::Vector<Real, 3> LinearMapping::map_to_physical(const math::Vector<Real, 3>& xi) const {
    std::vector<Real> N;
    basis_->evaluate_values(xi, N);
    math::Vector<Real, 3> x{};
    for (std::size_t a = 0; a < N.size(); ++a) {
        x += nodes_[a] * N[a];
    }
    return x;
}

math::Matrix<Real, 3, 3> LinearMapping::jacobian(const math::Vector<Real, 3>&) const {
    // For affine simplex elements gradients are constant; evaluate at a safe interior
    // point to avoid division by zero in barycentric derivatives.
    math::Vector<Real, 3> safe_xi{};
    if (dimension_ == 2) {
        safe_xi = math::Vector<Real, 3>{Real(1.0/3.0), Real(1.0/3.0), Real(0)};
    } else if (dimension_ == 3) {
        safe_xi = math::Vector<Real, 3>{Real(0.25), Real(0.25), Real(0.25)};
    }

    std::vector<basis::Gradient> grads;
    basis_->evaluate_gradients(safe_xi, grads);

    math::Matrix<Real, 3, 3> J{};
    for (std::size_t a = 0; a < grads.size(); ++a) {
        for (int j = 0; j < dimension_; ++j) {
            const std::size_t sj = static_cast<std::size_t>(j);
            for (std::size_t i = 0; i < 3; ++i) {
                J(i, sj) += nodes_[a][i] * grads[a][sj];
            }
        }
    }

    // Complete a 3x3 frame Jacobian for embedded curves/surfaces.
    if (dimension_ == 1) {
        const math::Vector<Real, 3> t{J(0, 0), J(1, 0), J(2, 0)};
        math::Vector<Real, 3> n1{};
        math::Vector<Real, 3> n2{};
        detail::complete_curve_frame(t, n1, n2);
        J(0, 1) = n1[0]; J(1, 1) = n1[1]; J(2, 1) = n1[2];
        J(0, 2) = n2[0]; J(1, 2) = n2[1]; J(2, 2) = n2[2];
    } else if (dimension_ == 2) {
        const math::Vector<Real, 3> tu{J(0, 0), J(1, 0), J(2, 0)};
        const math::Vector<Real, 3> tv{J(0, 1), J(1, 1), J(2, 1)};
        const auto n = tu.cross(tv);
        const Real n_norm = n.norm();
        if (n_norm < kDegenerateTol) {
            J(0, 2) = Real(0); J(1, 2) = Real(0); J(2, 2) = Real(0);
        } else {
            const auto n_unit = n / n_norm;
            J(0, 2) = n_unit[0]; J(1, 2) = n_unit[1]; J(2, 2) = n_unit[2];
        }
    }
    return J;
}

math::Vector<Real, 3> LinearMapping::map_to_reference(const math::Vector<Real, 3>& x_phys,
                                                      const math::Vector<Real, 3>& initial_guess) const {
    return InverseMapping::solve(*this, x_phys, initial_guess);
}

} // namespace geometry
} // namespace FE
} // namespace svmp
