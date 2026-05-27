/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "BatchedProjection.h"

#include "Math/SIMD.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace assembly {

void weighted_sum(const basis::BatchEvaluator& batch,
                  const Real* coeffs,
                  const Real* weights,
                  Real* result) {
    const auto& data = batch.data();
    const std::size_t num_quad = data.num_quad_points;
    const std::size_t num_basis = data.num_basis;

    using ops = math::simd::SIMDOps<Real>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    std::fill_n(result, num_quad, Real(0));

    if constexpr (vec_size > 1) {
        const std::size_t vec_end = num_quad - (num_quad % vec_size);

        for (std::size_t i = 0; i < num_basis; ++i) {
            const Real coeff = coeffs[i];
            vec_t vcoeff = ops::broadcast(coeff);
            const Real* basis_vals = data.values_for_basis(i);

            for (std::size_t q = 0; q < vec_end; q += vec_size) {
                vec_t vbasis = ops::load(basis_vals + q);
                vec_t vweight = ops::loadu(weights + q);
                vec_t vresult = ops::loadu(result + q);
                vresult = ops::fma(vcoeff, ops::mul(vbasis, vweight), vresult);
                ops::storeu(result + q, vresult);
            }

            for (std::size_t q = vec_end; q < num_quad; ++q) {
                result[q] += coeff * data.value(i, q) * weights[q];
            }
        }
        return;
    }

    for (std::size_t i = 0; i < num_basis; ++i) {
        const Real coeff = coeffs[i];
        for (std::size_t q = 0; q < num_quad; ++q) {
            result[q] += coeff * data.value(i, q) * weights[q];
        }
    }
}

void weighted_gradient_sum(const basis::BatchEvaluator& batch,
                           const Real* coeffs,
                           const Real* weights,
                           Real* result) {
    const auto& data = batch.data();
    if (!data.has_gradients) {
        throw basis::BasisEvaluationException(
            "weighted_gradient_sum: gradients not available in BatchEvaluator",
            __FILE__, __LINE__, __func__);
    }

    const std::size_t num_quad = data.num_quad_points;
    const std::size_t num_basis = data.num_basis;
    const int dim = batch.dimension();

    using ops = math::simd::SIMDOps<Real>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    std::fill_n(result, 3u * num_quad, Real(0));

    if constexpr (vec_size > 1) {
        const std::size_t vec_end = num_quad - (num_quad % vec_size);

        for (std::size_t i = 0; i < num_basis; ++i) {
            const Real coeff = coeffs[i];
            vec_t vcoeff = ops::broadcast(coeff);

            for (int d = 0; d < dim; ++d) {
                const auto sd = static_cast<std::size_t>(d);
                const Real* grad_vals = data.gradients_for_basis(i, sd);
                Real* result_d = result + sd * num_quad;

                for (std::size_t q = 0; q < vec_end; q += vec_size) {
                    vec_t vgrad = ops::load(grad_vals + q);
                    vec_t vweight = ops::loadu(weights + q);
                    vec_t vresult = ops::loadu(result_d + q);
                    vresult = ops::fma(vcoeff, ops::mul(vgrad, vweight), vresult);
                    ops::storeu(result_d + q, vresult);
                }

                for (std::size_t q = vec_end; q < num_quad; ++q) {
                    result_d[q] += coeff * data.gradient(i, sd, q) * weights[q];
                }
            }
        }
        return;
    }

    for (std::size_t i = 0; i < num_basis; ++i) {
        const Real coeff = coeffs[i];
        for (int d = 0; d < dim; ++d) {
            const auto sd = static_cast<std::size_t>(d);
            Real* result_d = result + sd * num_quad;
            for (std::size_t q = 0; q < num_quad; ++q) {
                result_d[q] += coeff * data.gradient(i, sd, q) * weights[q];
            }
        }
    }
}

} // namespace assembly
} // namespace FE
} // namespace svmp
