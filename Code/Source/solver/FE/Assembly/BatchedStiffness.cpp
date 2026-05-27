/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "BatchedStiffness.h"

#include "Math/SIMD.h"

namespace svmp {
namespace FE {
namespace assembly {

void assemble_stiffness_contribution(const basis::BatchEvaluator& batch,
                                     const Real* D,
                                     const Real* weights,
                                     Real* K) {
    const auto& data = batch.data();
    if (!data.has_gradients) {
        throw basis::BasisEvaluationException(
            "assemble_stiffness_contribution: gradients not available in BatchEvaluator",
            __FILE__, __LINE__, __func__);
    }

    const std::size_t num_quad = data.num_quad_points;
    const std::size_t num_basis = data.num_basis;
    const int dim = batch.dimension();

    using ops = math::simd::SIMDOps<Real>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    if constexpr (vec_size > 1) {
        const std::size_t vec_end = num_quad - (num_quad % vec_size);
        const std::size_t padded_quad = data.quad_stride;

        for (std::size_t i = 0; i < num_basis; ++i) {
            for (std::size_t j = 0; j < num_basis; ++j) {
                vec_t kij_vec = ops::zero();

                for (std::size_t q = 0; q < vec_end; q += vec_size) {
                    vec_t vweight = ops::loadu(weights + q);
                    vec_t contribution = ops::zero();

                    for (int d1 = 0; d1 < dim; ++d1) {
                        const Real* grad_i_d1 = data.gradients.data() +
                            (i * 3u + static_cast<std::size_t>(d1)) * padded_quad;
                        vec_t vgrad_i = ops::load(grad_i_d1 + q);

                        for (int d2 = 0; d2 < dim; ++d2) {
                            const Real* grad_j_d2 = data.gradients.data() +
                                (j * 3u + static_cast<std::size_t>(d2)) * padded_quad;
                            vec_t vgrad_j = ops::load(grad_j_d2 + q);
                            const Real D_val = D[static_cast<std::size_t>(d1 * dim + d2)];
                            vec_t vD = ops::broadcast(D_val);
                            contribution = ops::fma(ops::mul(vgrad_i, vD), vgrad_j, contribution);
                        }
                    }

                    kij_vec = ops::fma(contribution, vweight, kij_vec);
                }

                Real kij = ops::horizontal_sum(kij_vec);
                for (std::size_t q = vec_end; q < num_quad; ++q) {
                    Real contribution = Real(0);
                    for (int d1 = 0; d1 < dim; ++d1) {
                        for (int d2 = 0; d2 < dim; ++d2) {
                            contribution += data.gradient(i, static_cast<std::size_t>(d1), q) *
                                            D[static_cast<std::size_t>(d1 * dim + d2)] *
                                            data.gradient(j, static_cast<std::size_t>(d2), q);
                        }
                    }
                    kij += contribution * weights[q];
                }

                K[i * num_basis + j] = kij;
            }
        }
        return;
    }

    for (std::size_t i = 0; i < num_basis; ++i) {
        for (std::size_t j = 0; j < num_basis; ++j) {
            Real kij = Real(0);
            for (std::size_t q = 0; q < num_quad; ++q) {
                Real contribution = Real(0);
                for (int d1 = 0; d1 < dim; ++d1) {
                    for (int d2 = 0; d2 < dim; ++d2) {
                        contribution += data.gradient(i, static_cast<std::size_t>(d1), q) *
                                        D[static_cast<std::size_t>(d1 * dim + d2)] *
                                        data.gradient(j, static_cast<std::size_t>(d2), q);
                    }
                }
                kij += contribution * weights[q];
            }
            K[i * num_basis + j] = kij;
        }
    }
}

} // namespace assembly
} // namespace FE
} // namespace svmp
