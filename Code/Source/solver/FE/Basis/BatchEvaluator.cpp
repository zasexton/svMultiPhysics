/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "BatchEvaluator.h"

#include "Math/SIMD.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace basis {

namespace {

void resize_without_value_fill(AlignedRealVector& buffer, std::size_t size) {
    buffer.clear();
    buffer.resize(size);
}

void zero_tail_columns(AlignedRealVector& buffer,
                       std::size_t rows,
                       std::size_t logical_cols,
                       std::size_t row_stride) {
    if (logical_cols >= row_stride) {
        return;
    }

    for (std::size_t row = 0; row < rows; ++row) {
        Real* tail = buffer.data() + row * row_stride + logical_cols;
        std::fill(tail, tail + (row_stride - logical_cols), Real(0));
    }
}

} // namespace

BatchEvaluator::BatchEvaluator(const BasisFunction& basis,
                               const quadrature::QuadratureRule& quad,
                               bool compute_gradients,
                               bool compute_hessians)
    : dimension_(basis.dimension()) {
    if (basis.is_vector_valued()) {
        throw BasisConfigurationException("BatchEvaluator supports scalar bases only",
                                          __FILE__, __LINE__, __func__);
    }

    const std::size_t num_basis = basis.size();
    const std::size_t num_quad = quad.num_points();
    constexpr std::size_t simd_width = math::simd::SIMDCapabilities::double_width();
    const std::size_t padded_quad = ((num_quad + simd_width - 1) / simd_width) * simd_width;

    data_.num_basis = num_basis;
    data_.num_quad_points = num_quad;
    data_.quad_stride = padded_quad;
    data_.has_gradients = compute_gradients;
    data_.has_hessians = compute_hessians;

    // Allocate padded, aligned SoA storage without clearing logical entries;
    // strided evaluation overwrites them, and only SIMD tail columns need zero.
    resize_without_value_fill(data_.values, num_basis * padded_quad);
    if (compute_gradients) {
        resize_without_value_fill(data_.gradients, num_basis * 3u * padded_quad);
    }
    if (compute_hessians) {
        resize_without_value_fill(data_.hessians, num_basis * 9u * padded_quad);
    }

    if (num_basis > 0 && num_quad > 0) {
        basis.evaluate_at_quadrature_points_strided(
            quad.points(),
            padded_quad,
            data_.values.data(),
            compute_gradients ? data_.gradients.data() : nullptr,
            compute_hessians ? data_.hessians.data() : nullptr);
    }

    zero_tail_columns(data_.values, num_basis, num_quad, padded_quad);
    if (compute_gradients) {
        zero_tail_columns(data_.gradients, num_basis * 3u, num_quad, padded_quad);
    }
    if (compute_hessians) {
        zero_tail_columns(data_.hessians, num_basis * 9u, num_quad, padded_quad);
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
