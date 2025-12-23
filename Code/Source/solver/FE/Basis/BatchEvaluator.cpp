/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "BatchEvaluator.h"
#include <cstring>

namespace svmp {
namespace FE {
namespace basis {

BatchEvaluator::BatchEvaluator(const BasisFunction& basis,
                               const quadrature::QuadratureRule& quad,
                               bool compute_gradients,
                               bool compute_hessians)
    : dimension_(basis.dimension()) {
    if (basis.is_vector_valued()) {
        throw FEException("BatchEvaluator supports scalar bases only",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const std::size_t num_basis = basis.size();
    const std::size_t num_quad = quad.num_points();

    data_.num_basis = num_basis;
    data_.num_quad_points = num_quad;
    data_.has_gradients = compute_gradients;
    data_.has_hessians = compute_hessians;

    // Allocate storage in SoA format
    data_.values.resize(num_basis * num_quad);
    if (compute_gradients) {
        data_.gradients.resize(num_basis * 3 * num_quad);
    }
    if (compute_hessians) {
        data_.hessians.resize(num_basis * 9 * num_quad);
    }

    // Temporary storage for point-wise evaluation
    std::vector<Real> values_tmp(num_basis);
    std::vector<Gradient> gradients_tmp(num_basis);
    std::vector<Hessian> hessians_tmp(num_basis);

    // Evaluate at each quadrature point and transpose to SoA
    for (std::size_t q = 0; q < num_quad; ++q) {
        const auto& pt = quad.point(q);

        // Evaluate values
        basis.evaluate_values(pt, values_tmp);
        for (std::size_t i = 0; i < num_basis; ++i) {
            data_.values[i * num_quad + q] = values_tmp[i];
        }

        // Evaluate gradients
        if (compute_gradients) {
            basis.evaluate_gradients(pt, gradients_tmp);
            for (std::size_t i = 0; i < num_basis; ++i) {
                for (int d = 0; d < 3; ++d) {
                    data_.gradients[(i * 3 + static_cast<std::size_t>(d)) * num_quad + q] =
                        gradients_tmp[i][static_cast<std::size_t>(d)];
                }
            }
        }

        // Evaluate Hessians
        if (compute_hessians) {
            basis.evaluate_hessians(pt, hessians_tmp);
            for (std::size_t i = 0; i < num_basis; ++i) {
                for (int d1 = 0; d1 < 3; ++d1) {
                    for (int d2 = 0; d2 < 3; ++d2) {
                        data_.hessians[(i * 9 + static_cast<std::size_t>(d1 * 3 + d2)) * num_quad + q] =
                            hessians_tmp[i](static_cast<std::size_t>(d1), static_cast<std::size_t>(d2));
                    }
                }
            }
        }
    }

    // Create aligned copies for SIMD operations
    // Pad to SIMD width for efficient vectorization
    constexpr std::size_t simd_width = math::simd::SIMDCapabilities::double_width();
    const std::size_t padded_quad = ((num_quad + simd_width - 1) / simd_width) * simd_width;

    aligned_values_.resize(num_basis * padded_quad, Real(0));
    for (std::size_t i = 0; i < num_basis; ++i) {
        std::memcpy(aligned_values_.data() + i * padded_quad,
                    data_.values.data() + i * num_quad,
                    num_quad * sizeof(Real));
    }

    if (compute_gradients) {
        aligned_gradients_.resize(num_basis * 3 * padded_quad, Real(0));
        for (std::size_t i = 0; i < num_basis; ++i) {
            for (std::size_t d = 0; d < 3; ++d) {
                std::memcpy(aligned_gradients_.data() + (i * 3 + d) * padded_quad,
                            data_.gradients.data() + (i * 3 + d) * num_quad,
                            num_quad * sizeof(Real));
            }
        }
    }
}

void BatchEvaluator::weighted_sum(const Real* coeffs,
                                  const Real* weights,
                                  Real* result) const {
    const std::size_t num_quad = data_.num_quad_points;
    const std::size_t num_basis = data_.num_basis;

    using ops = math::simd::SIMDOps<Real>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    // Initialize result to zero
    std::memset(result, 0, num_quad * sizeof(Real));

    if constexpr (vec_size > 1) {
        // SIMD path
        const std::size_t vec_end = num_quad - (num_quad % vec_size);

        for (std::size_t i = 0; i < num_basis; ++i) {
            const Real coeff = coeffs[i];
            vec_t vcoeff = ops::broadcast(coeff);
            const Real* basis_vals = aligned_values_.data() + i * ((num_quad + vec_size - 1) / vec_size) * vec_size;

            // Vectorized portion
            for (std::size_t q = 0; q < vec_end; q += vec_size) {
                vec_t vbasis = ops::load(basis_vals + q);
                vec_t vweight = ops::loadu(weights + q);
                vec_t vresult = ops::loadu(result + q);
                vresult = ops::fma(vcoeff, ops::mul(vbasis, vweight), vresult);
                ops::storeu(result + q, vresult);
            }

            // Scalar tail
            for (std::size_t q = vec_end; q < num_quad; ++q) {
                result[q] += coeff * data_.value(i, q) * weights[q];
            }
        }
    } else {
        // Scalar fallback
        for (std::size_t i = 0; i < num_basis; ++i) {
            const Real coeff = coeffs[i];
            for (std::size_t q = 0; q < num_quad; ++q) {
                result[q] += coeff * data_.value(i, q) * weights[q];
            }
        }
    }
}

void BatchEvaluator::weighted_gradient_sum(const Real* coeffs,
                                           const Real* weights,
                                           Real* result) const {
    if (!data_.has_gradients) {
        throw FEException("BatchEvaluator: gradients not available (constructed with compute_gradients=false)",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const std::size_t num_quad = data_.num_quad_points;
    const std::size_t num_basis = data_.num_basis;
    const int dim = dimension_;

    using ops = math::simd::SIMDOps<Real>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    // Initialize result to zero (3 * num_quad)
    std::memset(result, 0, 3 * num_quad * sizeof(Real));

    if constexpr (vec_size > 1) {
        const std::size_t vec_end = num_quad - (num_quad % vec_size);
        const std::size_t padded_quad = ((num_quad + vec_size - 1) / vec_size) * vec_size;

        for (std::size_t i = 0; i < num_basis; ++i) {
            const Real coeff = coeffs[i];
            vec_t vcoeff = ops::broadcast(coeff);

            for (int d = 0; d < dim; ++d) {
                const Real* grad_vals = aligned_gradients_.data() + (i * 3 + static_cast<std::size_t>(d)) * padded_quad;
                Real* result_d = result + static_cast<std::size_t>(d) * num_quad;

                for (std::size_t q = 0; q < vec_end; q += vec_size) {
                    vec_t vgrad = ops::load(grad_vals + q);
                    vec_t vweight = ops::loadu(weights + q);
                    vec_t vresult = ops::loadu(result_d + q);
                    vresult = ops::fma(vcoeff, ops::mul(vgrad, vweight), vresult);
                    ops::storeu(result_d + q, vresult);
                }

                for (std::size_t q = vec_end; q < num_quad; ++q) {
                    result_d[q] += coeff * data_.gradient(i, static_cast<std::size_t>(d), q) * weights[q];
                }
            }
        }
    } else {
        // Scalar fallback
        for (std::size_t i = 0; i < num_basis; ++i) {
            const Real coeff = coeffs[i];
            for (int d = 0; d < dim; ++d) {
                Real* result_d = result + static_cast<std::size_t>(d) * num_quad;
                for (std::size_t q = 0; q < num_quad; ++q) {
                    result_d[q] += coeff * data_.gradient(i, static_cast<std::size_t>(d), q) * weights[q];
                }
            }
        }
    }
}

void BatchEvaluator::assemble_stiffness_contribution(const Real* D,
                                                     const Real* weights,
                                                     Real* K) const {
    if (!data_.has_gradients) {
        throw FEException("BatchEvaluator: gradients not available (constructed with compute_gradients=false)",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const std::size_t num_quad = data_.num_quad_points;
    const std::size_t num_basis = data_.num_basis;
    const int dim = dimension_;

    using ops = math::simd::SIMDOps<Real>;
    using vec_t = typename ops::vec_type;
    constexpr std::size_t vec_size = ops::vec_size;

    // K_ij = sum_q w_q * (sum_{d1,d2} dN_i/d(xi_{d1}) * D_{d1,d2} * dN_j/d(xi_{d2}))
    // Optimized for symmetric K: only compute upper triangle

    if constexpr (vec_size > 1) {
        const std::size_t vec_end = num_quad - (num_quad % vec_size);
        const std::size_t padded_quad = ((num_quad + vec_size - 1) / vec_size) * vec_size;

        for (std::size_t i = 0; i < num_basis; ++i) {
            for (std::size_t j = i; j < num_basis; ++j) {
                vec_t kij_vec = ops::zero();

                // Vectorized integration over quadrature points
                for (std::size_t q = 0; q < vec_end; q += vec_size) {
                    vec_t vweight = ops::loadu(weights + q);
                    vec_t contribution = ops::zero();

                    for (int d1 = 0; d1 < dim; ++d1) {
                        const Real* grad_i_d1 = aligned_gradients_.data() +
                            (i * 3 + static_cast<std::size_t>(d1)) * padded_quad;
                        vec_t vgrad_i = ops::load(grad_i_d1 + q);

                        for (int d2 = 0; d2 < dim; ++d2) {
                            const Real* grad_j_d2 = aligned_gradients_.data() +
                                (j * 3 + static_cast<std::size_t>(d2)) * padded_quad;
                            vec_t vgrad_j = ops::load(grad_j_d2 + q);

                            Real D_val = D[static_cast<std::size_t>(d1 * dim + d2)];
                            vec_t vD = ops::broadcast(D_val);

                            // contribution += grad_i * D * grad_j
                            contribution = ops::fma(ops::mul(vgrad_i, vD), vgrad_j, contribution);
                        }
                    }

                    kij_vec = ops::fma(contribution, vweight, kij_vec);
                }

                Real kij = ops::horizontal_sum(kij_vec);

                // Scalar tail
                for (std::size_t q = vec_end; q < num_quad; ++q) {
                    Real contribution = Real(0);
                    for (int d1 = 0; d1 < dim; ++d1) {
                        for (int d2 = 0; d2 < dim; ++d2) {
                            contribution += data_.gradient(i, static_cast<std::size_t>(d1), q) *
                                          D[static_cast<std::size_t>(d1 * dim + d2)] *
                                          data_.gradient(j, static_cast<std::size_t>(d2), q);
                        }
                    }
                    kij += contribution * weights[q];
                }

                K[i * num_basis + j] = kij;
                if (i != j) {
                    K[j * num_basis + i] = kij;  // Symmetry
                }
            }
        }
    } else {
        // Scalar fallback
        for (std::size_t i = 0; i < num_basis; ++i) {
            for (std::size_t j = i; j < num_basis; ++j) {
                Real kij = Real(0);
                for (std::size_t q = 0; q < num_quad; ++q) {
                    Real contribution = Real(0);
                    for (int d1 = 0; d1 < dim; ++d1) {
                        for (int d2 = 0; d2 < dim; ++d2) {
                            contribution += data_.gradient(i, static_cast<std::size_t>(d1), q) *
                                          D[static_cast<std::size_t>(d1 * dim + d2)] *
                                          data_.gradient(j, static_cast<std::size_t>(d2), q);
                        }
                    }
                    kij += contribution * weights[q];
                }
                K[i * num_basis + j] = kij;
                if (i != j) {
                    K[j * num_basis + i] = kij;
                }
            }
        }
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
