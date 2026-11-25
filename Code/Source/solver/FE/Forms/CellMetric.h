/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_FORMS_CELL_METRIC_H
#define SVMP_FE_FORMS_CELL_METRIC_H

#include "Forms/FormExpr.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace svmp::FE::forms {

/// Physical inverse-length metric for a cell in the given ambient dimension.
/// For a simplex, the reference metric M is the Gram matrix of an equal-volume
/// regular simplex expressed in the basis's reference coordinates. A wedge uses
/// the triangular metric in its base and unit axial scale. The line, square,
/// cube and square-based pyramid use the identity in their reference coordinates.
/// At each quadrature point the form evaluates J^{-T} M J^{-1}. For an embedded
/// cell J includes the mapping's orthonormal normal frame. Padding M with zeros
/// on that frame gives an ambient tensor whose null space is the normal space.
[[nodiscard]] inline FormExpr cellMetric(ElementType reference_cell, int ambient_dimension)
{
    const int intrinsic_dimension = element_dimension(reference_cell);
    if (intrinsic_dimension < 1) {
        throw std::invalid_argument("cellMetric requires a positive-dimensional reference cell");
    }
    if (ambient_dimension < intrinsic_dimension || ambient_dimension > 3) {
        throw std::invalid_argument("cellMetric ambient dimension must be between the cell dimension and three");
    }
    const auto inverse = FormExpr::jacobianInverse();
    const auto family = to_mesh_family(reference_cell);
    const bool has_simplex_factor = family == CellFamily::Triangle || family == CellFamily::Tetra ||
                                    family == CellFamily::Wedge;
    if (!has_simplex_factor && intrinsic_dimension == ambient_dimension) {
        return transpose(inverse) * inverse;
    }

    const auto zero = FormExpr::constant(0.0);
    const auto one = FormExpr::constant(1.0);
    std::vector<std::vector<FormExpr>> reference(ambient_dimension, std::vector<FormExpr>(ambient_dimension, zero));
    for (int i = 0; i < intrinsic_dimension; ++i) {
        reference[i][i] = one;
    }
    if (has_simplex_factor) {
        const int simplex_dimension = family == CellFamily::Tetra ? 3 : 2;
        const Real scale = simplex_dimension == 2 ? 1.0 / std::sqrt(3.0) : 1.0 / std::cbrt(4.0);
        const auto diagonal = FormExpr::constant(2.0 * scale);
        const auto off = FormExpr::constant(scale);
        for (int i = 0; i < simplex_dimension; ++i) {
            for (int j = 0; j < simplex_dimension; ++j) {
                reference[i][j] = i == j ? diagonal : off;
            }
        }
    }
    return transpose(inverse) * FormExpr::asTensor(std::move(reference)) * inverse;
}

/// Metric for a full-dimensional cell in its intrinsic spatial dimension.
[[nodiscard]] inline FormExpr cellMetric(ElementType reference_cell)
{
    return cellMetric(reference_cell, element_dimension(reference_cell));
}

} // namespace svmp::FE::forms

#endif // SVMP_FE_FORMS_CELL_METRIC_H
