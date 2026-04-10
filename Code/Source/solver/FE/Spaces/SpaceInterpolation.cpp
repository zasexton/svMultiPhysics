/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/SpaceInterpolation.h"

namespace svmp {
namespace FE {
namespace spaces {

namespace {

FunctionSpace::Value identity_point_map(const FunctionSpace::Value& xi) {
    return xi;
}

FunctionSpace::Value identity_value_transform(const FunctionSpace::Value& value) {
    return value;
}

} // namespace

void SpaceInterpolation::l2_projection(const FunctionSpace& src_space,
                                       const std::vector<Real>& src_coeffs,
                                       const FunctionSpace& dst_space,
                                       std::vector<Real>& dst_coeffs) {
    // Define reference-space function via source evaluation
    FunctionSpace::ValueFunction f = [&src_space, &src_coeffs](const FunctionSpace::Value& xi) {
        return src_space.evaluate(xi, src_coeffs);
    };
    dst_space.interpolate(f, dst_coeffs);
}

void SpaceInterpolation::nodal_interpolation(const FunctionSpace& src_space,
                                             const std::vector<Real>& src_coeffs,
                                             const FunctionSpace& dst_space,
                                             std::vector<Real>& dst_coeffs) {
    FE_CHECK_ARG(src_space.field_type() == FieldType::Scalar,
                 "SpaceInterpolation::nodal_interpolation: source must be scalar");
    FE_CHECK_ARG(dst_space.field_type() == FieldType::Scalar,
                 "SpaceInterpolation::nodal_interpolation: destination must be scalar");

    // Attempt to access LagrangeBasis for both spaces
    const auto& src_basis = src_space.element().basis();
    const auto& dst_basis = dst_space.element().basis();

    auto src_lagrange = dynamic_cast<const basis::LagrangeBasis*>(&src_basis);
    auto dst_lagrange = dynamic_cast<const basis::LagrangeBasis*>(&dst_basis);
    FE_CHECK_ARG(src_lagrange != nullptr && dst_lagrange != nullptr,
                 "SpaceInterpolation::nodal_interpolation: requires LagrangeBasis");

    const auto& dst_nodes = dst_lagrange->nodes();
    dst_coeffs.resize(dst_nodes.size());

    // For scalar Lagrange spaces, DOFs are nodal values
    for (std::size_t i = 0; i < dst_nodes.size(); ++i) {
        const auto& xi = dst_nodes[i];
        const auto val = src_space.evaluate(xi, src_coeffs);
        dst_coeffs[i] = val[0];
    }
}

SpaceInterpolation::TransferOperator SpaceInterpolation::build_transfer_operator(
    const FunctionSpace& src_space,
    const FunctionSpace& dst_space,
    const ReferencePointMap& point_map,
    const ValueTransform& value_transform) {
    const std::size_t src_dofs = src_space.dofs_per_element();
    const std::size_t dst_dofs = dst_space.dofs_per_element();

    TransferOperator op;
    op.rows = dst_dofs;
    op.cols = src_dofs;
    op.values.assign(dst_dofs * src_dofs, Real(0));

    if (src_dofs == 0 || dst_dofs == 0) {
        return op;
    }

    const auto& point_pullback = point_map ? point_map : ReferencePointMap(identity_point_map);
    const auto& value_pullback =
        value_transform ? value_transform : ValueTransform(identity_value_transform);

    std::vector<Real> src_coeffs(src_dofs, Real(0));
    std::vector<Real> dst_column;
    for (std::size_t col = 0; col < src_dofs; ++col) {
        std::fill(src_coeffs.begin(), src_coeffs.end(), Real(0));
        src_coeffs[col] = Real(1);

        FunctionSpace::ValueFunction lifted =
            [&src_space, &src_coeffs, &point_pullback, &value_pullback](const FunctionSpace::Value& xi_dst) {
                const auto xi_src = point_pullback(xi_dst);
                return value_pullback(src_space.evaluate(xi_src, src_coeffs));
            };

        dst_space.interpolate(lifted, dst_column);
        FE_CHECK_ARG(dst_column.size() == dst_dofs,
                     "SpaceInterpolation::build_transfer_operator: destination interpolation size mismatch");
        for (std::size_t row = 0; row < dst_dofs; ++row) {
            op(row, col) = dst_column[row];
        }
    }

    return op;
}

void SpaceInterpolation::apply_transfer(const TransferOperator& op,
                                        std::span<const Real> src_coeffs,
                                        std::vector<Real>& dst_coeffs) {
    FE_CHECK_ARG(src_coeffs.size() == op.cols,
                 "SpaceInterpolation::apply_transfer: source coefficient size mismatch");

    dst_coeffs.assign(op.rows, Real(0));
    for (std::size_t row = 0; row < op.rows; ++row) {
        Real sum = Real(0);
        for (std::size_t col = 0; col < op.cols; ++col) {
            sum += op(row, col) * src_coeffs[col];
        }
        dst_coeffs[row] = sum;
    }
}

void SpaceInterpolation::prolongate(const FunctionSpace& coarse_space,
                                    const std::vector<Real>& coarse_coeffs,
                                    const FunctionSpace& fine_space,
                                    std::vector<Real>& fine_coeffs,
                                    const ReferencePointMap& point_map,
                                    const ValueTransform& value_transform) {
    const auto op = prolongation_operator(coarse_space, fine_space, point_map, value_transform);
    SpaceInterpolation::apply_transfer(op, coarse_coeffs, fine_coeffs);
}

void SpaceInterpolation::restrict_coefficients(const FunctionSpace& fine_space,
                                               const std::vector<Real>& fine_coeffs,
                                               const FunctionSpace& coarse_space,
                                               std::vector<Real>& coarse_coeffs,
                                               const ReferencePointMap& point_map,
                                               const ValueTransform& value_transform) {
    const auto op = restriction_operator(fine_space, coarse_space, point_map, value_transform);
    SpaceInterpolation::apply_transfer(op, fine_coeffs, coarse_coeffs);
}

} // namespace spaces
} // namespace FE
} // namespace svmp
