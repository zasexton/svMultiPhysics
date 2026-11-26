/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "IsoparametricMapping.h"
#include "InverseMapping.h"
#include "Core/FEException.h"
#include "Basis/LagrangeBasis.h"
#include "Basis/SerendipityBasis.h"
#include "GeometryFrameUtils.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <typeinfo>

namespace svmp {
namespace FE {
namespace geometry {

namespace {

// For these nodal bases, agreement of every geometry coefficient with an affine
// map proves equality throughout the cell by unisolvence and linear precision.
const std::vector<math::Vector<Real, 3>>* interpolationNodes(const basis::BasisFunction& basis)
{
    if (typeid(basis) == typeid(basis::LagrangeBasis)) {
        return &static_cast<const basis::LagrangeBasis&>(basis).nodes();
    }
    if (typeid(basis) == typeid(basis::SerendipityBasis)) {
        return &static_cast<const basis::SerendipityBasis&>(basis).nodes();
    }
    return nullptr;
}

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
    if (nodes_.empty()) {
        return {};
    }

    // Geometry bases form a partition of unity.  Accumulating relative to an
    // anchor node preserves that invariant numerically and avoids injecting a
    // large global-coordinate offset into the interpolation sum.
    const auto& anchor = nodes_.front();
    math::Vector<Real, 3> x = anchor;
    for (std::size_t a = 0; a < N.size(); ++a) {
        x += (nodes_[a] - anchor) * N[a];
    }
    return x;
}

math::Matrix<Real, 3, 3> IsoparametricMapping::jacobian(const math::Vector<Real, 3>& xi) const {
    std::vector<basis::Gradient> grads;
    basis_->evaluate_gradients(xi, grads);
    math::Matrix<Real, 3, 3> J{};
    const int dim = basis_->dimension();
    const math::Vector<Real, 3> anchor =
        nodes_.empty() ? math::Vector<Real, 3>{} : nodes_.front();
    for (std::size_t a = 0; a < grads.size(); ++a) {
        for (int j = 0; j < dim; ++j) {
            const std::size_t sj = static_cast<std::size_t>(j);
            for (std::size_t i = 0; i < 3; ++i) {
                J(i, sj) += (nodes_[a][i] - anchor[i]) * grads[a][sj];
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
        math::Vector<Real, 3> n_unit{};
        if (!detail::complete_surface_frame(tu, tv, n_unit)) {
            // Degenerate surface: keep third column zero so det(J)=0.
            J(0, 2) = Real(0); J(1, 2) = Real(0); J(2, 2) = Real(0);
        } else {
            J(0, 2) = n_unit[0]; J(1, 2) = n_unit[1]; J(2, 2) = n_unit[2];
        }
    }
    return J;
}

GeometryMapping::MappingHessian IsoparametricMapping::mapping_hessian(const math::Vector<Real, 3>& xi) const
{
    std::vector<basis::Hessian> hessians;
    basis_->evaluate_hessians(xi, hessians);

    MappingHessian H{};
    const int dim = basis_->dimension();
    const math::Vector<Real, 3> anchor =
        nodes_.empty() ? math::Vector<Real, 3>{} : nodes_.front();

    for (std::size_t a = 0; a < hessians.size(); ++a) {
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                const Real d2N = hessians[a](static_cast<std::size_t>(i), static_cast<std::size_t>(j));
                for (std::size_t m = 0; m < 3; ++m) {
                    H[m](static_cast<std::size_t>(i), static_cast<std::size_t>(j)) +=
                        (nodes_[a][m] - anchor[m]) * d2N;
                }
            }
        }
    }

    return H;
}

bool IsoparametricMapping::isAffine() const noexcept
{
    try {
        const auto* reference = interpolationNodes(*basis_);
        if (!reference || reference->empty() || reference->size() != nodes_.size()) {
            return false;
        }
        math::Vector<Real, 3> center{};
        for (const auto& point : *reference) {
            center += point;
        }
        center /= static_cast<Real>(reference->size());
        const auto linear = jacobian(center);
        math::Vector<Real, 3> span{}, error{};
        for (std::size_t node = 0; node < nodes_.size(); ++node) {
            const math::Vector<Real, 3> delta = (*reference)[node] - reference->front();
            const auto displacement = nodes_[node] - nodes_.front();
            const auto predicted = linear * delta;
            for (std::size_t component = 0; component < 3; ++component) {
                if (!std::isfinite(displacement[component]) || !std::isfinite(predicted[component])) {
                    return false;
                }
                span[component] = std::max(span[component], std::abs(displacement[component]));
                error[component] = std::max(error[component],
                                            std::abs(displacement[component] - predicted[component]));
            }
        }
        constexpr Real roundoff = Real(32) * std::numeric_limits<Real>::epsilon();
        for (std::size_t component = 0; component < 3; ++component) {
            if (error[component] > roundoff * span[component]) {
                return false;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

math::Vector<Real, 3> IsoparametricMapping::map_to_reference(const math::Vector<Real, 3>& x_phys,
                                                            const math::Vector<Real, 3>& initial_guess) const {
    return InverseMapping::solve(*this, x_phys, initial_guess);
}

} // namespace geometry
} // namespace FE
} // namespace svmp
