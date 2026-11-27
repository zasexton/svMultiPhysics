/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/SpaceInterpolation.h"
#include "Math/DenseLinearAlgebra.h"
#include "Quadrature/CompositeQuadrature.h"
#include "Quadrature/QuadratureFactory.h"

#include <algorithm>
#include <iterator>
#include <utility>

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

// Unit coefficient vectors expose the represented basis of a product or
// enriched space as well as that of an ordinary scalar or vector element.
void evaluate_space_basis(const FunctionSpace& space, const FunctionSpace::Value& point,
                          std::vector<Real>& unit, std::vector<FunctionSpace::Value>& values) {
    for (std::size_t i = 0; i < unit.size(); ++i) {
        unit[i] = 1.;
        values[i] = space.evaluate(point, unit);
        unit[i] = 0.;
    }
}

std::shared_ptr<const quadrature::QuadratureRule> restriction_quadrature(
    const FunctionSpace& fine, const FunctionSpace& coarse, bool shared_coordinates) {
    const auto src = fine.element().quadrature();
    const auto dst = coarse.element().quadrature();
    const int order = std::max(fine.polynomial_order(), coarse.polynomial_order());
    const int exactness = std::max({quadrature::QuadratureFactory::recommended_order(order, true),
                                    src->order(), dst->order()});
    auto base = quadrature::QuadratureFactory::create(coarse.element_type(), exactness);
    using Partition = quadrature::TensorPartitionQuadrature;
    const auto* dst_partition = dynamic_cast<const Partition*>(dst.get());
    const auto* src_partition = shared_coordinates && src->cell_family() == dst->cell_family()
        ? dynamic_cast<const Partition*>(src.get()) : nullptr;
    if (!dst_partition && !src_partition) return base;

    auto breaks = (dst_partition ? dst_partition : src_partition)->axis_breaks();
    if (dst_partition && src_partition) {
        for (std::size_t d = 0; d < breaks.size(); ++d) {
            const auto& other = src_partition->axis_breaks()[d];
            std::vector<Real> merged;
            std::set_union(breaks[d].begin(), breaks[d].end(), other.begin(), other.end(),
                           std::back_inserter(merged));
            breaks[d] = std::move(merged);
        }
    }
    return std::make_shared<Partition>(*base, std::move(breaks));
}

} // namespace

void SpaceInterpolation::l2_projection(const FunctionSpace& src_space,
                                       const std::vector<Real>& src_coeffs,
                                       const FunctionSpace& dst_space,
                                       std::vector<Real>& dst_coeffs) {
    restrict_coefficients(src_space, src_coeffs, dst_space, dst_coeffs);
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

SpaceInterpolation::TransferOperator SpaceInterpolation::restriction_operator(
    const FunctionSpace& fine_space, const FunctionSpace& coarse_space,
    const ReferencePointMap& point_map, const ValueTransform& value_transform) {
    const auto rows = coarse_space.dofs_per_element();
    const auto cols = fine_space.dofs_per_element();
    TransferOperator cross_mass{rows, cols, std::vector<Real>(rows * cols, 0.)};
    if (!rows || !cols) return cross_mass;

    const auto rule = restriction_quadrature(fine_space, coarse_space, !point_map);
    std::vector<Real> mass(rows * rows, 0.), coarse_unit(rows, 0.), fine_unit(cols, 0.);
    std::vector<FunctionSpace::Value> coarse(rows), fine(cols);
    for (std::size_t q = 0; q < rule->num_points(); ++q) {
        const auto point = rule->point(q);
        evaluate_space_basis(coarse_space, point, coarse_unit, coarse);
        evaluate_space_basis(fine_space, point_map ? point_map(point) : point, fine_unit, fine);
        if (value_transform) {
            for (auto& value : fine) value = value_transform(value);
        }
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < rows; ++j)
                mass[i * rows + j] += rule->weight(q) * coarse[i].dot(coarse[j]);
            for (std::size_t j = 0; j < cols; ++j)
                cross_mass(i, j) += rule->weight(q) * coarse[i].dot(fine[j]);
        }
    }
    const auto solver = math::factor_dense_matrix(std::move(mass), rows, "reference L2 restriction");
    solver.solve_in_place(cross_mass.values, cols);
    return cross_mass;
}

void SpaceInterpolation::apply_transfer(const TransferOperator& op,
                                        std::span<const Real> src_coeffs,
                                        std::vector<Real>& dst_coeffs) {
    FE_CHECK_ARG(src_coeffs.size() == op.cols,
                 "SpaceInterpolation::apply_transfer: source coefficient size mismatch");

    std::vector<Real> transferred(op.rows, Real(0));
    for (std::size_t row = 0; row < op.rows; ++row) {
        Real sum = Real(0);
        for (std::size_t col = 0; col < op.cols; ++col) {
            sum += op(row, col) * src_coeffs[col];
        }
        transferred[row] = sum;
    }
    dst_coeffs = std::move(transferred);
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
