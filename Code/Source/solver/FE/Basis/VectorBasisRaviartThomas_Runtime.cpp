/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "VectorBasis.h"
#include "Basis/BasisTraits.h"
#include "VectorBasisDirectSeeds.h"
#include "VectorBasisRtConstruction.h"

#include <cmath>
#include <cstddef>
#include <vector>

#ifdef FE_CHECK_ARG
#undef FE_CHECK_ARG
#endif
#define FE_CHECK_ARG(condition, message) BASIS_CHECK_CONSTRUCTION((condition), (message))

#include "VectorBasisEvaluationHelpers.h"

namespace svmp {
namespace FE {
namespace basis {

using namespace detail::vector_common;

using detail::vector_construction::eval_rt_seed_divergence;
using detail::vector_construction::eval_rt_seed_values;

// -----------------------------------------------------------------------------
// Pyramid RT0 helper (H(div))
// -----------------------------------------------------------------------------

namespace {

constexpr Real kTetraRt0OppositeVertexScale =
    Real(-1.15470053837925152901829756100391491129520350254025L); // -2/sqrt(3)

// Rational RT0 on pyramid (H(div)) – implemented via CAS-derived polynomials.
inline void eval_pyramid_rational_rt0(const math::Vector<Real, 3>& xi,
                                      std::vector<math::Vector<Real, 3>>& values) {
    // Polynomial RT0 basis on the reference pyramid (Pyramid5) with one
    // face-flux DOF per face. Coordinates (x,y,z) follow LagrangeBasis:
    // base z=0 square (-1,-1,0)..(1,1,0), apex at (0,0,1).
    const Real x = xi[0];
    const Real y = xi[1];
    values.resize(5);
    values[0] = math::Vector<Real, 3>{Real(3) * x / Real(8),
                                      Real(3) * y / Real(8),
                                      Real(-1) / Real(4)};
    values[1] = math::Vector<Real, 3>{Real(0),
                                      Real(3) * y / Real(4) - Real(1) / Real(2),
                                      Real(0)};
    values[2] = math::Vector<Real, 3>{Real(3) * x / Real(4) + Real(1) / Real(2),
                                      Real(0),
                                      Real(0)};
    values[3] = math::Vector<Real, 3>{Real(0),
                                      Real(3) * y / Real(4) + Real(1) / Real(2),
                                      Real(0)};
    values[4] = math::Vector<Real, 3>{Real(3) * x / Real(4) - Real(1) / Real(2),
                                      Real(0),
                                      Real(0)};
}

inline void eval_pyramid_rational_rt0_divergence(std::vector<Real>& divergence) {
    divergence.assign(5, Real(3) / Real(4));
}

} // namespace

void RaviartThomasBasis::evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                                std::vector<math::Vector<Real, 3>>& values) const {
    if (use_transformed_direct_seed_) {
        const std::size_t n = size_;
        const std::size_t num_seed = transformed_seed_indices_.size();
        const std::size_t num_extra = transformed_monomial_candidates_.size();
        const std::size_t candidate_count = num_seed + num_extra;
        values.assign(n, Vec3{});
        FE_CHECK_ARG(transformed_sparse_coeffs_.rows == candidate_count &&
                         transformed_sparse_coeffs_.cols == n &&
                         transformed_sparse_coeffs_.row_offsets.size() == candidate_count + 1u,
                     "RaviartThomasBasis::evaluate_vector_values: transformed RT sparse coefficient size mismatch");

        auto add_candidate_value = [&](std::size_t candidate, const Vec3& value) {
            const std::size_t row_begin = transformed_sparse_coeffs_.row_offsets[candidate];
            const std::size_t row_end = transformed_sparse_coeffs_.row_offsets[candidate + 1u];
            for (std::size_t entry = row_begin; entry < row_end; ++entry) {
                const std::size_t dof = transformed_sparse_coeffs_.dofs[entry];
                const Real c = transformed_sparse_coeffs_.coefficients[entry];
                values[dof][0] += c * value[0];
                values[dof][1] += c * value[1];
                values[dof][2] += c * value[2];
            }
        };

        std::size_t candidate = 0;
        if (num_seed > 0) {
            auto& seed_values = vector_basis_scratch().vector_values;
            eval_rt_seed_values(element_type_, order_, xi, seed_values);
            FE_CHECK_ARG(seed_values.size() >= num_seed,
                         "RaviartThomasBasis::evaluate_vector_values: RT seed basis size mismatch");

            for (int seed_idx : transformed_seed_indices_) {
                FE_CHECK_ARG(seed_idx >= 0 &&
                                 static_cast<std::size_t>(seed_idx) < seed_values.size(),
                             "RaviartThomasBasis::evaluate_vector_values: transformed RT seed index out of range");
                const Vec3& seed = seed_values[static_cast<std::size_t>(seed_idx)];
                add_candidate_value(candidate, seed);
                ++candidate;
            }
        }

        if (num_extra > 0) {
            auto& scratch = vector_basis_scratch();
            fill_power_tables(xi, transformed_power_limits_, scratch);
            const auto& px = scratch.px;
            const auto& py = scratch.py;
            const auto& pz = scratch.pz;

            for (const auto& mono : transformed_monomial_candidates_) {
                const Real scalar = eval_transformed_rt_monomial_scalar(mono, px, py, pz);
                if (scalar != Real(0)) {
                    const auto component = static_cast<std::size_t>(mono[0]);
                    const std::size_t row_begin = transformed_sparse_coeffs_.row_offsets[candidate];
                    const std::size_t row_end = transformed_sparse_coeffs_.row_offsets[candidate + 1u];
                    for (std::size_t entry = row_begin; entry < row_end; ++entry) {
                        const std::size_t dof = transformed_sparse_coeffs_.dofs[entry];
                        values[dof][component] +=
                            transformed_sparse_coeffs_.coefficients[entry] * scalar;
                    }
                }
                ++candidate;
            }
        }
        return;
    }

    if (nodal_generated_) {
        evaluate_nodal_modal_vector_values_with_limits(
            monomials_, modal_sparse_coeffs_, size_, xi, modal_power_limits_, values);
        return;
    }

    if (dimension_ == 2) {
        const Real x = xi[0];
        const Real y = xi[1];

        if (is_triangle(element_type_)) {
            // Minimal RT0 on reference triangle (v0=(0,0), v1=(1,0), v2=(0,1)).
            // Basis functions chosen so that integrated normal flux over each edge
            // (with standard outward normals) gives Kronecker delta DOFs.
            constexpr Real inv_sqrt2 = Real(0.70710678118654752440084436210484903928483593768847L); // 1/sqrt(2)
            values.resize(3);
            values[0] = math::Vector<Real, 3>{inv_sqrt2 * x, inv_sqrt2 * y, Real(0)};
            values[1] = math::Vector<Real, 3>{x - Real(1), y, Real(0)};
            values[2] = math::Vector<Real, 3>{x, y - Real(1), Real(0)};
            return;
        }

        // Quadrilateral RT0 on [-1,1]^2
        values.resize(4);
        values[0] = math::Vector<Real, 3>{Real(0.5) * (Real(1) + x), Real(0), Real(0)};
        values[1] = math::Vector<Real, 3>{Real(0.5) * (Real(1) - x), Real(0), Real(0)};
        values[2] = math::Vector<Real, 3>{Real(0), Real(0.5) * (Real(1) + y), Real(0)};
        values[3] = math::Vector<Real, 3>{Real(0), Real(0.5) * (Real(1) - y), Real(0)};
        return;
    }

    const Real x = xi[0];
    const Real y = xi[1];
    const Real z = xi[2];

    if (is_tetrahedron(element_type_)) {
        // Minimal RT0 on reference tetra (v0=(0,0,0), v1=(1,0,0),
        // v2=(0,1,0), v3=(0,0,1)) with DOFs as face fluxes.
        // Constructed so that ∫_{Fi} v_j · n_i dS = δ_{ij} using
        // n_i = grad(lambda_i) and standard face parameterizations.
        values.resize(4);
        // Ordering matches ReferenceElement (Tetra4) face list:
        //   f0: (0,1,2) opposite v3
        //   f1: (0,1,3) opposite v2
        //   f2: (1,2,3) opposite v0
        //   f3: (0,2,3) opposite v1
        values[0] = math::Vector<Real, 3>{-Real(2) * x,
                                          -Real(2) * y,
                                          Real(2) - Real(2) * z};
        values[1] = math::Vector<Real, 3>{-Real(2) * x,
                                          Real(2) - Real(2) * y,
                                          -Real(2) * z};
        values[2] = math::Vector<Real, 3>{kTetraRt0OppositeVertexScale * x,
                                          kTetraRt0OppositeVertexScale * y,
                                          kTetraRt0OppositeVertexScale * z};
        values[3] = math::Vector<Real, 3>{Real(2) - Real(2) * x,
                                          -Real(2) * y,
                                          -Real(2) * z};
        return;
    }

    if (is_wedge(element_type_)) {
        // Minimal RT0 on wedge6: five basis functions, one per face.
        values.resize(5);
        values[0] = math::Vector<Real, 3>{Real(0), Real(0), z - Real(1)}; // bottom face
        values[1] = math::Vector<Real, 3>{Real(0), Real(0), z + Real(1)}; // top face
        values[2] = math::Vector<Real, 3>{x, -Real(0.5), Real(0)};        // y=0 face
        values[3] = math::Vector<Real, 3>{Real(0.5) - x, Real(0), Real(0)}; // x=0 face
        values[4] = math::Vector<Real, 3>{-x, Real(0), Real(0)};          // x+y=1 face
        return;
    }

    if (is_pyramid(element_type_)) {
        eval_pyramid_rational_rt0(xi, values);
        return;
    }

    // Hexahedron / wedge: simple face-aligned RT0-like fields
    values.resize(6);
    values[0] = math::Vector<Real, 3>{Real(0.25) * (Real(1) + x), Real(0), Real(0)};
    values[1] = math::Vector<Real, 3>{Real(0.25) * (Real(1) - x), Real(0), Real(0)};
    values[2] = math::Vector<Real, 3>{Real(0), Real(0.25) * (Real(1) + y), Real(0)};
    values[3] = math::Vector<Real, 3>{Real(0), Real(0.25) * (Real(1) - y), Real(0)};
    values[4] = math::Vector<Real, 3>{Real(0), Real(0), Real(0.25) * (Real(1) + z)};
    values[5] = math::Vector<Real, 3>{Real(0), Real(0), Real(0.25) * (Real(1) - z)};
}

void RaviartThomasBasis::evaluate_vector_jacobians(const math::Vector<Real, 3>& xi,
                                                   std::vector<VectorJacobian>& jacobians) const {
    if (use_transformed_direct_seed_) {
        const std::size_t n = size_;
        const std::size_t num_seed = transformed_seed_indices_.size();
        const std::size_t num_extra = transformed_monomial_candidates_.size();
        const std::size_t candidate_count = num_seed + num_extra;
        jacobians.assign(n, VectorJacobian{});
        FE_CHECK_ARG(transformed_sparse_coeffs_.rows == candidate_count &&
                         transformed_sparse_coeffs_.cols == n &&
                         transformed_sparse_coeffs_.row_offsets.size() == candidate_count + 1u,
                     "RaviartThomasBasis::evaluate_vector_jacobians: transformed RT sparse coefficient size mismatch");

        auto add_candidate_jacobian = [&](std::size_t candidate, const VectorJacobian& seed) {
            const std::size_t row_begin = transformed_sparse_coeffs_.row_offsets[candidate];
            const std::size_t row_end = transformed_sparse_coeffs_.row_offsets[candidate + 1u];
            for (std::size_t entry = row_begin; entry < row_end; ++entry) {
                const std::size_t dof = transformed_sparse_coeffs_.dofs[entry];
                const Real c = transformed_sparse_coeffs_.coefficients[entry];
                for (std::size_t r = 0; r < 3u; ++r) {
                    for (std::size_t col = 0; col < 3u; ++col) {
                        jacobians[dof](r, col) += c * seed(r, col);
                    }
                }
            }
        };

        std::size_t candidate = 0;
        if (num_seed > 0) {
            auto& seed_jacobians = vector_basis_scratch().vector_jacobians;
            if (is_wedge(element_type_) && order_ == 1) {
                detail::vector_direct::eval_wedge_rt1_jacobians(xi, seed_jacobians);
            } else if (is_wedge(element_type_) && order_ == 2) {
                detail::vector_direct::eval_wedge_rt2_jacobians(xi, seed_jacobians);
            } else if (is_pyramid(element_type_) && order_ == 1) {
                detail::vector_direct::eval_pyramid_rt1_jacobians(xi, seed_jacobians);
            } else if (is_pyramid(element_type_) && order_ == 2) {
                detail::vector_direct::eval_pyramid_rt2_jacobians(xi, seed_jacobians);
            } else {
                throw NotImplementedException(
                    "RaviartThomasBasis::evaluate_vector_jacobians: transformed RT seed Jacobians currently support wedge/pyramid orders 1-2",
                    __FILE__, __LINE__, __func__);
            }
            FE_CHECK_ARG(seed_jacobians.size() >= num_seed,
                         "RaviartThomasBasis::evaluate_vector_jacobians: RT seed Jacobian size mismatch");

            for (int seed_idx : transformed_seed_indices_) {
                FE_CHECK_ARG(seed_idx >= 0 &&
                                 static_cast<std::size_t>(seed_idx) < seed_jacobians.size(),
                             "RaviartThomasBasis::evaluate_vector_jacobians: transformed RT seed index out of range");
                const auto& seed = seed_jacobians[static_cast<std::size_t>(seed_idx)];
                add_candidate_jacobian(candidate, seed);
                ++candidate;
            }
        }

        if (num_extra > 0) {
            auto& scratch = vector_basis_scratch();
            fill_power_tables(xi, transformed_power_limits_, scratch);
            const auto& px = scratch.px;
            const auto& py = scratch.py;
            const auto& pz = scratch.pz;

            for (const auto& mono : transformed_monomial_candidates_) {
                const std::size_t row_begin = transformed_sparse_coeffs_.row_offsets[candidate];
                const std::size_t row_end = transformed_sparse_coeffs_.row_offsets[candidate + 1u];
                for (std::size_t entry = row_begin; entry < row_end; ++entry) {
                    const std::size_t dof = transformed_sparse_coeffs_.dofs[entry];
                    add_component_monomial_jacobian(
                        jacobians[dof],
                        mono[0],
                        mono[1],
                        mono[2],
                        mono[3],
                        transformed_sparse_coeffs_.coefficients[entry],
                        px,
                        py,
                        pz);
                }
                ++candidate;
            }
        }
        return;
    }

    if (nodal_generated_) {
        evaluate_nodal_modal_vector_jacobians_with_limits(
            monomials_, modal_sparse_coeffs_, size_, xi, modal_power_limits_, jacobians);
        return;
    }

    jacobians.assign(size_, VectorJacobian{});

    if (dimension_ == 2) {
        if (is_triangle(element_type_)) {
            constexpr Real inv_sqrt2 = Real(0.70710678118654752440084436210484903928483593768847L);
            jacobians[0](0, 0) = inv_sqrt2;
            jacobians[0](1, 1) = inv_sqrt2;
            jacobians[1](0, 0) = Real(1);
            jacobians[1](1, 1) = Real(1);
            jacobians[2](0, 0) = Real(1);
            jacobians[2](1, 1) = Real(1);
            return;
        }
        jacobians[0](0, 0) = Real(0.5);
        jacobians[1](0, 0) = -Real(0.5);
        jacobians[2](1, 1) = Real(0.5);
        jacobians[3](1, 1) = -Real(0.5);
        return;
    }

    if (is_tetrahedron(element_type_)) {
        jacobians[0](0, 0) = -Real(2);
        jacobians[0](1, 1) = -Real(2);
        jacobians[0](2, 2) = -Real(2);
        jacobians[1](0, 0) = -Real(2);
        jacobians[1](1, 1) = -Real(2);
        jacobians[1](2, 2) = -Real(2);
        jacobians[2](0, 0) = kTetraRt0OppositeVertexScale;
        jacobians[2](1, 1) = kTetraRt0OppositeVertexScale;
        jacobians[2](2, 2) = kTetraRt0OppositeVertexScale;
        jacobians[3](0, 0) = -Real(2);
        jacobians[3](1, 1) = -Real(2);
        jacobians[3](2, 2) = -Real(2);
        return;
    }

    if (is_wedge(element_type_)) {
        jacobians[0](2, 2) = Real(1);
        jacobians[1](2, 2) = Real(1);
        jacobians[2](0, 0) = Real(1);
        jacobians[3](0, 0) = -Real(1);
        jacobians[4](0, 0) = -Real(1);
        return;
    }

    if (is_pyramid(element_type_)) {
        jacobians[0](0, 0) = Real(3) / Real(8);
        jacobians[0](1, 1) = Real(3) / Real(8);
        jacobians[1](1, 1) = Real(3) / Real(4);
        jacobians[2](0, 0) = Real(3) / Real(4);
        jacobians[3](1, 1) = Real(3) / Real(4);
        jacobians[4](0, 0) = Real(3) / Real(4);
        return;
    }

    jacobians[0](0, 0) = Real(0.25);
    jacobians[1](0, 0) = -Real(0.25);
    jacobians[2](1, 1) = Real(0.25);
    jacobians[3](1, 1) = -Real(0.25);
    jacobians[4](2, 2) = Real(0.25);
    jacobians[5](2, 2) = -Real(0.25);
}

void RaviartThomasBasis::evaluate_divergence(const math::Vector<Real, 3>& xi,
                                             std::vector<Real>& divergence) const {
    if (use_transformed_direct_seed_) {
        const std::size_t n = size_;
        const std::size_t num_seed = transformed_seed_indices_.size();
        const std::size_t num_extra = transformed_monomial_candidates_.size();
        const std::size_t candidate_count = num_seed + num_extra;
        divergence.assign(n, Real(0));
        FE_CHECK_ARG(transformed_sparse_coeffs_.rows == candidate_count &&
                         transformed_sparse_coeffs_.cols == n &&
                         transformed_sparse_coeffs_.row_offsets.size() == candidate_count + 1u,
                     "RaviartThomasBasis::evaluate_divergence: transformed RT sparse coefficient size mismatch");

        std::size_t candidate = 0;
        if (num_seed > 0) {
            auto& seed_divergence = vector_basis_scratch().scalars;
            eval_rt_seed_divergence(element_type_, order_, xi, seed_divergence);
            FE_CHECK_ARG(seed_divergence.size() >= num_seed,
                         "RaviartThomasBasis::evaluate_divergence: RT seed divergence size mismatch");

            for (int seed_idx : transformed_seed_indices_) {
                FE_CHECK_ARG(seed_idx >= 0 &&
                                 static_cast<std::size_t>(seed_idx) < seed_divergence.size(),
                             "RaviartThomasBasis::evaluate_divergence: transformed RT seed index out of range");
                const Real seed = seed_divergence[static_cast<std::size_t>(seed_idx)];
                if (seed == Real(0)) {
                    ++candidate;
                    continue;
                }
                const std::size_t row_begin = transformed_sparse_coeffs_.row_offsets[candidate];
                const std::size_t row_end = transformed_sparse_coeffs_.row_offsets[candidate + 1u];
                for (std::size_t entry = row_begin; entry < row_end; ++entry) {
                    divergence[transformed_sparse_coeffs_.dofs[entry]] +=
                        transformed_sparse_coeffs_.coefficients[entry] * seed;
                }
                ++candidate;
            }
        }

        if (num_extra > 0) {
            auto& scratch = vector_basis_scratch();
            fill_power_tables(xi, transformed_power_limits_, scratch);
            const auto& px = scratch.px;
            const auto& py = scratch.py;
            const auto& pz = scratch.pz;

            for (const auto& mono : transformed_monomial_candidates_) {
                const Real div = eval_transformed_rt_monomial_divergence(mono, px, py, pz);
                if (div == Real(0)) {
                    ++candidate;
                    continue;
                }
                const std::size_t row_begin = transformed_sparse_coeffs_.row_offsets[candidate];
                const std::size_t row_end = transformed_sparse_coeffs_.row_offsets[candidate + 1u];
                for (std::size_t entry = row_begin; entry < row_end; ++entry) {
                    divergence[transformed_sparse_coeffs_.dofs[entry]] +=
                        transformed_sparse_coeffs_.coefficients[entry] * div;
                }
                ++candidate;
            }
        }
        return;
    }

    if (nodal_generated_) {
        evaluate_nodal_modal_divergence_with_limits(
            monomials_, modal_sparse_coeffs_, size_, xi, modal_power_limits_, divergence);
        return;
    }

    if (dimension_ == 2) {
        if (is_triangle(element_type_)) {
            // Constant divergences corresponding to the minimal RT0 triangle basis above.
            const Real inv_sqrt2 = Real(1.0 / std::sqrt(2.0));
            divergence = {inv_sqrt2 * Real(2), Real(2), Real(2)};
        } else {
            divergence = {Real(0.5), Real(-0.5), Real(0.5), Real(-0.5)};
        }
    } else {
        if (is_tetrahedron(element_type_)) {
            // Constant divergences corresponding to the minimal RT0 tetra basis above.
            // Keep ordering consistent with evaluate_vector_values (ReferenceElement face order).
            divergence = {Real(-6),
                          Real(-6),
                          Real(3) * kTetraRt0OppositeVertexScale,
                          Real(-6)};
        } else if (is_wedge(element_type_)) {
            // Divergences of the minimal RT0 wedge basis
            divergence = {Real(1), Real(1), Real(1), Real(-1), Real(-1)};
        } else if (is_pyramid(element_type_)) {
            eval_pyramid_rational_rt0_divergence(divergence);
        } else {
            // Hexahedra: face-aligned RT0-like fields
            divergence = {Real(0.25), Real(-0.25), Real(0.25), Real(-0.25), Real(0.25), Real(-0.25)};
        }
    }
}

void RaviartThomasBasis::evaluate_vector_at_quadrature_points_strided(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT jacobians_out,
    Real* SVMP_RESTRICT curls_out,
    Real* SVMP_RESTRICT divergence_out) const {
    if (nodal_generated_) {
        evaluate_nodal_modal_vector_strided_with_limits(
            monomials_,
            modal_sparse_coeffs_,
            size_,
            points,
            output_stride,
            modal_power_limits_,
            values_out,
            jacobians_out,
            curls_out,
            divergence_out,
            "RaviartThomasBasis");
        return;
    }

    if (use_transformed_direct_seed_) {
        const std::size_t num_qpts = points.size();
        const std::size_t n = size_;
        const std::size_t num_seed = transformed_seed_indices_.size();
        const std::size_t num_extra = transformed_monomial_candidates_.size();
        const std::size_t candidate_count = num_seed + num_extra;
        validate_vector_strided_outputs(num_qpts, output_stride, "RaviartThomasBasis");
        FE_CHECK_ARG(transformed_sparse_coeffs_.rows == candidate_count &&
                         transformed_sparse_coeffs_.cols == n &&
                         transformed_sparse_coeffs_.row_offsets.size() == candidate_count + 1u,
                     "RaviartThomasBasis strided transformed RT sparse coefficient size mismatch");

        auto& scratch = vector_basis_scratch();
        const bool need_values = values_out != nullptr;
        const bool need_jacobians = jacobians_out != nullptr;
        const bool need_curls = curls_out != nullptr;
        const bool need_divergence = divergence_out != nullptr;
        const bool need_derivative_tensor = need_jacobians || need_curls;
        if (!need_values && !need_jacobians && !need_curls && !need_divergence) {
            return;
        }

        if (need_values) {
            zero_active_strided_rows(values_out, n * 3u, output_stride, num_qpts);
        }
        if (need_jacobians) {
            zero_active_strided_rows(jacobians_out, n * 9u, output_stride, num_qpts);
        }
        if (need_curls) {
            zero_active_strided_rows(curls_out, n * 3u, output_stride, num_qpts);
        }
        if (need_divergence) {
            zero_active_strided_rows(divergence_out, n, output_stride, num_qpts);
        }

        if (num_seed > 0) {
            for (std::size_t q = 0; q < num_qpts; ++q) {
                std::size_t candidate = 0;
                if (need_values) {
                    eval_rt_seed_values(element_type_, order_, points[q], scratch.api_values);
                    FE_CHECK_ARG(scratch.api_values.size() >= num_seed,
                                 "RaviartThomasBasis strided RT seed value size mismatch");
                }
                if (need_derivative_tensor) {
                    FE_CHECK_ARG(transformed_seed_jacobian_evaluator_ != nullptr,
                                 "RaviartThomasBasis strided transformed RT seed Jacobian evaluator is not configured");
                    transformed_seed_jacobian_evaluator_(points[q], scratch.api_jacobians);
                    FE_CHECK_ARG(scratch.api_jacobians.size() >= num_seed,
                                 "RaviartThomasBasis strided RT seed Jacobian size mismatch");
                } else if (need_divergence) {
                    eval_rt_seed_divergence(element_type_, order_, points[q], scratch.api_divergence);
                    FE_CHECK_ARG(scratch.api_divergence.size() >= num_seed,
                                 "RaviartThomasBasis strided RT seed divergence size mismatch");
                }

                for (int seed_idx : transformed_seed_indices_) {
                    FE_CHECK_ARG(seed_idx >= 0 &&
                                     static_cast<std::size_t>(seed_idx) <
                                         (need_values ? scratch.api_values.size()
                                                      : need_derivative_tensor ? scratch.api_jacobians.size()
                                                                               : scratch.api_divergence.size()),
                                 "RaviartThomasBasis strided transformed RT seed index out of range");
                    const std::size_t seed = static_cast<std::size_t>(seed_idx);
                    const Vec3 seed_value = need_values ? scratch.api_values[seed] : Vec3{};
                    const VectorJacobian seed_jacobian =
                        need_derivative_tensor ? scratch.api_jacobians[seed] : VectorJacobian{};
                    const Vec3 seed_curl =
                        need_derivative_tensor ? curl_from_jacobian(seed_jacobian) : Vec3{};
                    const Real seed_divergence =
                        need_derivative_tensor ? divergence_from_jacobian(seed_jacobian)
                                               : need_divergence ? scratch.api_divergence[seed]
                                                                 : Real(0);

                    const std::size_t row_begin = transformed_sparse_coeffs_.row_offsets[candidate];
                    const std::size_t row_end = transformed_sparse_coeffs_.row_offsets[candidate + 1u];
                    for (std::size_t entry = row_begin; entry < row_end; ++entry) {
                        const std::size_t dof = transformed_sparse_coeffs_.dofs[entry];
                        const Real c = transformed_sparse_coeffs_.coefficients[entry];
                        if (need_values) {
                            for (std::size_t component = 0; component < 3u; ++component) {
                                values_out[(dof * 3u + component) * output_stride + q] +=
                                    c * seed_value[component];
                            }
                        }
                        if (need_jacobians) {
                            for (std::size_t row = 0; row < 3u; ++row) {
                                for (std::size_t col = 0; col < 3u; ++col) {
                                    jacobians_out[(dof * 9u + row * 3u + col) *
                                                      output_stride + q] +=
                                        c * seed_jacobian(row, col);
                                }
                            }
                        }
                        if (need_curls) {
                            for (std::size_t component = 0; component < 3u; ++component) {
                                curls_out[(dof * 3u + component) * output_stride + q] +=
                                    c * seed_curl[component];
                            }
                        }
                        if (need_divergence) {
                            divergence_out[dof * output_stride + q] += c * seed_divergence;
                        }
                    }
                    ++candidate;
                }
            }
        }

        if (num_extra > 0) {
            fill_batched_power_tables(points, transformed_power_limits_, scratch);
            const auto& px = scratch.batched_px;
            const auto& py = scratch.batched_py;
            const auto& pz = scratch.batched_pz;
            const bool need_modal_gradient =
                need_jacobians || need_curls || need_divergence;

            std::size_t candidate = num_seed;
            for (const auto& mono : transformed_monomial_candidates_) {
                const int component = mono[0];
                const int px_pow = mono[1];
                const int py_pow = mono[2];
                const int pz_pow = mono[3];
                const std::size_t component_index =
                    static_cast<std::size_t>(component);
                const std::size_t row_begin =
                    transformed_sparse_coeffs_.row_offsets[candidate];
                const std::size_t row_end =
                    transformed_sparse_coeffs_.row_offsets[candidate + 1u];
                auto& candidate_values = scratch.candidate_values;
                auto& candidate_dx = scratch.candidate_dx;
                auto& candidate_dy = scratch.candidate_dy;
                auto& candidate_dz = scratch.candidate_dz;

                if (need_values) {
                    candidate_values.resize(num_qpts);
                }
                if (need_modal_gradient) {
                    candidate_dx.resize(num_qpts);
                    candidate_dy.resize(num_qpts);
                    candidate_dz.resize(num_qpts);
                }

                for (std::size_t qp = 0; qp < num_qpts; ++qp) {
                    if (need_values) {
                        candidate_values[qp] =
                            batched_power_product(px,
                                                  py,
                                                  pz,
                                                  num_qpts,
                                                  px_pow,
                                                  py_pow,
                                                  pz_pow,
                                                  qp);
                    }
                    if (need_modal_gradient) {
                        candidate_dx[qp] =
                            batched_component_partial(px,
                                                      py,
                                                      pz,
                                                      num_qpts,
                                                      px_pow,
                                                      py_pow,
                                                      pz_pow,
                                                      0,
                                                      qp);
                        candidate_dy[qp] =
                            batched_component_partial(px,
                                                      py,
                                                      pz,
                                                      num_qpts,
                                                      px_pow,
                                                      py_pow,
                                                      pz_pow,
                                                      1,
                                                      qp);
                        candidate_dz[qp] =
                            batched_component_partial(px,
                                                      py,
                                                      pz,
                                                      num_qpts,
                                                      px_pow,
                                                      py_pow,
                                                      pz_pow,
                                                      2,
                                                      qp);
                    }
                }

                for (std::size_t entry = row_begin; entry < row_end; ++entry) {
                    const std::size_t dof = transformed_sparse_coeffs_.dofs[entry];
                    const Real coefficient =
                        transformed_sparse_coeffs_.coefficients[entry];
                    Real* value_row = need_values
                        ? values_out + (dof * 3u + component_index) * output_stride
                        : nullptr;
                    Real* jacobian_row = need_jacobians
                        ? jacobians_out +
                              (dof * 9u + component_index * 3u) * output_stride
                        : nullptr;
                    Real* curl_row = need_curls
                        ? curls_out + dof * 3u * output_stride
                        : nullptr;
                    Real* divergence_row = need_divergence
                        ? divergence_out + dof * output_stride
                        : nullptr;

                    for (std::size_t qp = 0; qp < num_qpts; ++qp) {
                        if (need_values) {
                            value_row[qp] += coefficient * candidate_values[qp];
                        }

                        if (need_modal_gradient) {
                            const Real dphidx = candidate_dx[qp];
                            const Real dphidy = candidate_dy[qp];
                            const Real dphidz = candidate_dz[qp];
                            if (need_jacobians) {
                                jacobian_row[qp] += coefficient * dphidx;
                                jacobian_row[output_stride + qp] += coefficient * dphidy;
                                jacobian_row[2u * output_stride + qp] +=
                                    coefficient * dphidz;
                            }
                            if (need_curls) {
                                const Vec3 curl =
                                    curl_from_component_gradient(component,
                                                                 dphidx,
                                                                 dphidy,
                                                                 dphidz);
                                for (std::size_t curl_component = 0;
                                     curl_component < 3u;
                                     ++curl_component) {
                                    curl_row[curl_component * output_stride + qp] +=
                                        coefficient * curl[curl_component];
                                }
                            }
                            if (need_divergence) {
                                const Real div = component == 0 ? dphidx
                                               : component == 1 ? dphidy
                                                                : dphidz;
                                divergence_row[qp] += coefficient * div;
                            }
                        }
                    }
                }
                ++candidate;
            }
        }
        return;
    }

    evaluate_vector_public_api_strided(*this,
                                       points,
                                       output_stride,
                                       values_out,
                                       jacobians_out,
                                       curls_out,
                                       divergence_out,
                                       false,
                                       true,
                                       "RaviartThomasBasis");
}

} // namespace basis
} // namespace FE
} // namespace svmp
