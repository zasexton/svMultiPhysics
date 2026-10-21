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
#include <array>
#include <cmath>

namespace svmp {
namespace FE {
namespace geometry {

namespace {

bool is_simplex(ElementType type) {
    return type == ElementType::Line2 || type == ElementType::Triangle3 || type == ElementType::Tetra4;
}

// The linear Lagrange basis of an affine simplex depends only on its type.
// Cut-cell measures, face frames and the free-surface snapshot construct a
// LinearMapping for every cell they evaluate (about 0.2 million per 3D sessile
// step at R/h = 8), so one immutable basis per type is built on first use
// (thread-safe initialization of a function-local static) and shared.  It is
// only evaluated through const member functions, which keep their scratch per
// thread, so concurrent mappings may share it.
std::shared_ptr<const basis::LagrangeBasis> sharedLinearBasis(ElementType type) {
    switch (type) {
    case ElementType::Line2: {
        static const auto shared = std::make_shared<const basis::LagrangeBasis>(ElementType::Line2, 1);
        return shared;
    }
    case ElementType::Triangle3: {
        static const auto shared = std::make_shared<const basis::LagrangeBasis>(ElementType::Triangle3, 1);
        return shared;
    }
    case ElementType::Tetra4: {
        static const auto shared = std::make_shared<const basis::LagrangeBasis>(ElementType::Tetra4, 1);
        return shared;
    }
    default:
        return std::make_shared<const basis::LagrangeBasis>(type, 1);
    }
}

// The interior point jacobian() evaluates the constant gradients at (barycentric
// derivatives are singular on the boundary).
math::Vector<Real, 3> safeReferencePoint(int dimension) {
    if (dimension == 2) {
        return math::Vector<Real, 3>{Real(1.0/3.0), Real(1.0/3.0), Real(0)};
    }
    if (dimension == 3) {
        return math::Vector<Real, 3>{Real(0.25), Real(0.25), Real(0.25)};
    }
    return math::Vector<Real, 3>{};
}

// The order-1 basis gradients at safeReferencePoint(), per simplex type.  They
// are computed once, by the shared basis at the same point jacobian() used to
// evaluate them on every call, so they are the same values.
const std::vector<basis::Gradient>* sharedSafeGradients(ElementType type) {
    const auto evaluate = [](ElementType t) {
        std::vector<basis::Gradient> gradients;
        sharedLinearBasis(t)->evaluate_gradients(safeReferencePoint(element_dimension(t)),
                                                 gradients);
        return gradients;
    };
    switch (type) {
    case ElementType::Line2: {
        static const auto gradients = evaluate(ElementType::Line2);
        return &gradients;
    }
    case ElementType::Triangle3: {
        static const auto gradients = evaluate(ElementType::Triangle3);
        return &gradients;
    }
    case ElementType::Tetra4: {
        static const auto gradients = evaluate(ElementType::Tetra4);
        return &gradients;
    }
    default:
        return nullptr;
    }
}

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
    basis_ = sharedLinearBasis(type);
    if (basis_->size() != nodes_.size()) {
        throw FEException("LinearMapping node count does not match basis size",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    safe_gradients_ = sharedSafeGradients(type);
    if (safe_gradients_ == nullptr || safe_gradients_->size() != basis_->size()) {
        throw FEException("LinearMapping has no constant gradient table for its element type",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }
}

math::Vector<Real, 3> LinearMapping::map_to_physical(const math::Vector<Real, 3>& xi) const {
    // At most four order-1 simplex basis values.  evaluate_values_to() runs the
    // same LagrangeLine/Tri/TetFast<1>::evaluate as evaluate_values(), without
    // the per-call vector.
    std::array<Real, 4> N{};
    const std::size_t n = basis_->size();
    basis_->evaluate_values_to(xi, N.data());
    math::Vector<Real, 3> x{};
    for (std::size_t a = 0; a < n; ++a) {
        x += nodes_[a] * N[a];
    }
    return x;
}

math::Matrix<Real, 3, 3> LinearMapping::jacobian(const math::Vector<Real, 3>&) const {
    // For affine simplex elements gradients are constant; they were evaluated
    // once, at a safe interior point (no division by zero in barycentric
    // derivatives), by sharedSafeGradients().
    const auto& grads = *safe_gradients_;

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
        math::Vector<Real, 3> n_unit{};
        if (!detail::complete_surface_frame(tu, tv, n_unit)) {
            J(0, 2) = Real(0); J(1, 2) = Real(0); J(2, 2) = Real(0);
        } else {
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
