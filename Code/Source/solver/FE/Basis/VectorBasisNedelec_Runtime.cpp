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
using detail::vector_construction::eval_nd_seed_curl;
using detail::vector_construction::eval_nd_seed_values;


void NedelecBasis::evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                          std::vector<math::Vector<Real, 3>>& values) const {
    if (use_transformed_direct_seed_) {
        const std::size_t n = size_;
        const std::size_t num_seed = (order_ <= 2) ? n : 0u;
        const std::size_t num_extra = transformed_monomial_candidates_.size();
        const std::size_t candidate_count = num_seed + num_extra;
        values.assign(n, math::Vector<Real, 3>{});

        FE_CHECK_ARG(transformed_sparse_coeffs_.rows == candidate_count &&
                         transformed_sparse_coeffs_.cols == n &&
                         transformed_sparse_coeffs_.row_offsets.size() == candidate_count + 1u,
                     "NedelecBasis::evaluate_vector_values: transformed ND sparse coefficient size mismatch");

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

        if (num_seed > 0) {
            auto& seed_values = vector_basis_scratch().vector_values;
            eval_nd_seed_values(element_type_, order_, xi, seed_values);
            FE_CHECK_ARG(seed_values.size() == n,
                         "NedelecBasis::evaluate_vector_values: ND seed basis size mismatch");

            for (std::size_t p = 0; p < n; ++p) {
                add_candidate_value(p, seed_values[p]);
            }
        }

        if (num_extra > 0) {
            auto& scratch = vector_basis_scratch();
            fill_power_tables(xi, transformed_power_limits_, scratch);
            const auto& px = scratch.px;
            const auto& py = scratch.py;
            const auto& pz = scratch.pz;

            std::size_t candidate = num_seed;
            for (const auto& mono : transformed_monomial_candidates_) {
                const Real scalar = eval_transformed_nd_monomial_scalar(mono, px, py, pz);
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
            // Simple edge-oriented Nedelec0-like fields on reference triangle
            values.resize(3);
            values[0] = math::Vector<Real, 3>{-y,           x,            Real(0)};
            values[1] = math::Vector<Real, 3>{-y,           x - Real(1),  Real(0)};
            values[2] = math::Vector<Real, 3>{Real(1) - y,  x,            Real(0)};
        } else {
            // Quadrilateral Nedelec0 on [-1,1]^2 with one edge DOF per edge:
            //   E_e(w) = ∫_{edge e} w · t_e ds, with edge orientation following
            //   the canonical Quad4 edge list: (0-1), (1-2), (2-3), (3-0).
            values.resize(4);
            values[0] = math::Vector<Real, 3>{ Real(0.25) * (Real(1) - y), Real(0), Real(0)};   // edge 0-1
            values[1] = math::Vector<Real, 3>{ Real(0), Real(0.25) * (Real(1) + x), Real(0)};   // edge 1-2
            values[2] = math::Vector<Real, 3>{ -Real(0.25) * (Real(1) + y), Real(0), Real(0)};  // edge 2-3
            values[3] = math::Vector<Real, 3>{ Real(0), -Real(0.25) * (Real(1) - x), Real(0)};  // edge 3-0
        }
        return;
    }

    const Real x = xi[0];
    const Real y = xi[1];
    const Real z = xi[2];

    if (is_wedge(element_type_)) {
        // Minimal Nedelec0 wedge basis: 9 edge-based functions with unit edge DOFs
        // on the reference wedge with vertices:
        //   v0=(0,0,-1), v1=(1,0,-1), v2=(0,1,-1),
        //   v3=(0,0,+1), v4=(1,0,+1), v5=(0,1,+1).
        //
        // Construction:
        //   - bottom/top horizontal edges: triangle Nedelec0 × linear z selector
        //   - vertical edges: vertex Lagrange × constant z-directed field
        values.resize(9);
        const Real lb = (Real(1) - z) * Real(0.5);
        const Real lt = (Real(1) + z) * Real(0.5);

        // Triangle Nedelec0 basis on (x,y) with vertices (0,0), (1,0), (0,1):
        //   edge 0-1: (1 - y, x)
        //   edge 1-2: (-y, x)
        //   edge 2-0: (-y, x - 1)
        values[0] = math::Vector<Real, 3>{(Real(1) - y) * lb, x * lb, Real(0)};          // bottom edge 0-1
        values[1] = math::Vector<Real, 3>{-y * lb, x * lb, Real(0)};                     // bottom edge 1-2
        values[2] = math::Vector<Real, 3>{-y * lb, (x - Real(1)) * lb, Real(0)};         // bottom edge 2-0

        values[3] = math::Vector<Real, 3>{(Real(1) - y) * lt, x * lt, Real(0)};          // top edge 3-4
        values[4] = math::Vector<Real, 3>{-y * lt, x * lt, Real(0)};                     // top edge 4-5
        values[5] = math::Vector<Real, 3>{-y * lt, (x - Real(1)) * lt, Real(0)};         // top edge 5-3

        // Vertical edges: v_z = 0.5 * λ_i(x,y), λ0=1-x-y, λ1=x, λ2=y.
        values[6] = math::Vector<Real, 3>{Real(0), Real(0), Real(0.5) * (Real(1) - x - y)}; // edge 0-3
        values[7] = math::Vector<Real, 3>{Real(0), Real(0), Real(0.5) * x};                 // edge 1-4
        values[8] = math::Vector<Real, 3>{Real(0), Real(0), Real(0.5) * y};                 // edge 2-5
        return;
    }

    if (is_pyramid(element_type_)) {
        // Minimal Nedelec0 basis on reference Pyramid5 (8 edges).
        values.resize(8);
        // Using the CAS-derived basis with vertices:
        // v0=(-1,-1,0), v1=(1,-1,0), v2=(1,1,0), v3=(-1,1,0), v4=(0,0,1).
        // Coordinates (x,y,z) follow LagrangeBasis.
        values[0] = math::Vector<Real, 3>{
            Real(1) / Real(4) - y * Real(1) / Real(4),
            Real(0),
            x * (Real(4) - Real(3) * y) / Real(8)
        };
        values[1] = math::Vector<Real, 3>{
            Real(0),
            x * Real(1) / Real(4) + Real(1) / Real(4),
            y * (Real(3) * x + Real(4)) / Real(8)
        };
        values[2] = math::Vector<Real, 3>{
            -y * Real(1) / Real(4) - Real(1) / Real(4),
            Real(0),
            x * (-Real(3) * y - Real(4)) / Real(8)
        };
        values[3] = math::Vector<Real, 3>{
            Real(0),
            x * Real(1) / Real(4) - Real(1) / Real(4),
            y * (Real(3) * x - Real(4)) / Real(8)
        };
        values[4] = math::Vector<Real, 3>{
            Real(0),
            Real(0),
            Real(3) * x * y / Real(4) - x * Real(1) / Real(2) - y * Real(1) / Real(2) + Real(1) / Real(4)
        };
        values[5] = math::Vector<Real, 3>{
            Real(0),
            Real(0),
            -Real(3) * x * y / Real(4) + x * Real(1) / Real(2) - y * Real(1) / Real(2) + Real(1) / Real(4)
        };
        values[6] = math::Vector<Real, 3>{
            Real(0),
            Real(0),
            Real(3) * x * y / Real(4) + x * Real(1) / Real(2) + y * Real(1) / Real(2) + Real(1) / Real(4)
        };
        values[7] = math::Vector<Real, 3>{
            Real(0),
            Real(0),
            -Real(3) * x * y / Real(4) - x * Real(1) / Real(2) + y * Real(1) / Real(2) + Real(1) / Real(4)
        };
        return;
    }

    if (is_tetrahedron(element_type_)) {
        // Nedelec0 on the reference tetrahedron with vertices
        // (0,0,0), (1,0,0), (0,1,0), (0,0,1).
        const Real lam0 = Real(1) - x - y - z;
        const Real lam1 = x;
        const Real lam2 = y;
        const Real lam3 = z;

        // Edge ordering matches ReferenceElement (Tetra4):
        // (0-1), (1-2), (2-0), (0-3), (1-3), (2-3).
        values.resize(6);
        values[0] = math::Vector<Real, 3>{lam0 + lam1, lam1, lam1};     // N_01
        values[1] = math::Vector<Real, 3>{-lam2, lam1, Real(0)};        // N_12
        values[2] = math::Vector<Real, 3>{-lam2, -lam2 - lam0, -lam2};  // N_20
        values[3] = math::Vector<Real, 3>{lam3, lam3, lam0 + lam3};     // N_03
        values[4] = math::Vector<Real, 3>{-lam3, Real(0), lam1};        // N_13
        values[5] = math::Vector<Real, 3>{Real(0), -lam3, lam2};        // N_23
        return;
    }

    // Nedelec0 on the reference hexahedron [-1,1]^3 (one edge DOF per edge).
    values.resize(12);
    // Bottom face edges (z=-1)
    values[0] = math::Vector<Real, 3>{ Real(0.125) * (Real(1) - y) * (Real(1) - z), Real(0), Real(0)};   // 0-1
    values[1] = math::Vector<Real, 3>{ Real(0), Real(0.125) * (Real(1) + x) * (Real(1) - z), Real(0)};   // 1-2
    values[2] = math::Vector<Real, 3>{ -Real(0.125) * (Real(1) + y) * (Real(1) - z), Real(0), Real(0)};  // 2-3
    values[3] = math::Vector<Real, 3>{ Real(0), -Real(0.125) * (Real(1) - x) * (Real(1) - z), Real(0)};  // 3-0
    // Top face edges (z=+1)
    values[4] = math::Vector<Real, 3>{ Real(0.125) * (Real(1) - y) * (Real(1) + z), Real(0), Real(0)};   // 4-5
    values[5] = math::Vector<Real, 3>{ Real(0), Real(0.125) * (Real(1) + x) * (Real(1) + z), Real(0)};   // 5-6
    values[6] = math::Vector<Real, 3>{ -Real(0.125) * (Real(1) + y) * (Real(1) + z), Real(0), Real(0)};  // 6-7
    values[7] = math::Vector<Real, 3>{ Real(0), -Real(0.125) * (Real(1) - x) * (Real(1) + z), Real(0)};  // 7-4
    // Vertical edges
    values[8]  = math::Vector<Real, 3>{ Real(0), Real(0), Real(0.125) * (Real(1) - x) * (Real(1) - y)};  // 0-4
    values[9]  = math::Vector<Real, 3>{ Real(0), Real(0), Real(0.125) * (Real(1) + x) * (Real(1) - y)};  // 1-5
    values[10] = math::Vector<Real, 3>{ Real(0), Real(0), Real(0.125) * (Real(1) + x) * (Real(1) + y)};  // 2-6
    values[11] = math::Vector<Real, 3>{ Real(0), Real(0), Real(0.125) * (Real(1) - x) * (Real(1) + y)};  // 3-7
}

void NedelecBasis::evaluate_vector_jacobians(const math::Vector<Real, 3>& xi,
                                             std::vector<VectorJacobian>& jacobians) const {
    if (use_transformed_direct_seed_) {
        const std::size_t n = size_;
        const std::size_t num_seed = (order_ <= 2) ? n : 0u;
        const std::size_t num_extra = transformed_monomial_candidates_.size();
        const std::size_t candidate_count = num_seed + num_extra;
        jacobians.assign(n, VectorJacobian{});

        FE_CHECK_ARG(transformed_sparse_coeffs_.rows == candidate_count &&
                         transformed_sparse_coeffs_.cols == n &&
                         transformed_sparse_coeffs_.row_offsets.size() == candidate_count + 1u,
                     "NedelecBasis::evaluate_vector_jacobians: transformed ND sparse coefficient size mismatch");

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

        if (num_seed > 0) {
            auto& seed_jacobians = vector_basis_scratch().vector_jacobians;
            if (is_wedge(element_type_) && order_ == 1) {
                detail::vector_direct::eval_wedge_nd1_jacobians(xi, seed_jacobians);
            } else if (is_wedge(element_type_) && order_ == 2) {
                detail::vector_direct::eval_wedge_nd2_jacobians(xi, seed_jacobians);
            } else if (is_pyramid(element_type_) && order_ == 1) {
                detail::vector_direct::eval_pyramid_nd1_jacobians(xi, seed_jacobians);
            } else if (is_pyramid(element_type_) && order_ == 2) {
                detail::vector_direct::eval_pyramid_nd2_jacobians(xi, seed_jacobians);
            } else {
                throw NotImplementedException(
                    "NedelecBasis::evaluate_vector_jacobians: transformed ND seed Jacobians currently support wedge/pyramid orders 1-2",
                    __FILE__, __LINE__, __func__);
            }
            FE_CHECK_ARG(seed_jacobians.size() == n,
                         "NedelecBasis::evaluate_vector_jacobians: ND seed Jacobian size mismatch");

            for (std::size_t p = 0; p < n; ++p) {
                add_candidate_jacobian(p, seed_jacobians[p]);
            }
        }

        if (num_extra > 0) {
            auto& scratch = vector_basis_scratch();
            fill_power_tables(xi, transformed_power_limits_, scratch);
            const auto& px = scratch.px;
            const auto& py = scratch.py;
            const auto& pz = scratch.pz;

            std::size_t candidate = num_seed;
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
            for (auto& J : jacobians) {
                J(0, 1) = -Real(1);
                J(1, 0) = Real(1);
            }
            return;
        }
        jacobians[0](0, 1) = -Real(0.25);
        jacobians[1](1, 0) = Real(0.25);
        jacobians[2](0, 1) = -Real(0.25);
        jacobians[3](1, 0) = Real(0.25);
        return;
    }

    const Real x = xi[0];
    const Real y = xi[1];
    const Real z = xi[2];

    if (is_wedge(element_type_)) {
        const Real lb = (Real(1) - z) * Real(0.5);
        const Real lt = (Real(1) + z) * Real(0.5);

        jacobians[0](0, 1) = -lb;
        jacobians[0](0, 2) = -(Real(1) - y) * Real(0.5);
        jacobians[0](1, 0) = lb;
        jacobians[0](1, 2) = -x * Real(0.5);

        jacobians[1](0, 1) = -lb;
        jacobians[1](0, 2) = y * Real(0.5);
        jacobians[1](1, 0) = lb;
        jacobians[1](1, 2) = -x * Real(0.5);

        jacobians[2](0, 1) = -lb;
        jacobians[2](0, 2) = y * Real(0.5);
        jacobians[2](1, 0) = lb;
        jacobians[2](1, 2) = -(x - Real(1)) * Real(0.5);

        jacobians[3](0, 1) = -lt;
        jacobians[3](0, 2) = (Real(1) - y) * Real(0.5);
        jacobians[3](1, 0) = lt;
        jacobians[3](1, 2) = x * Real(0.5);

        jacobians[4](0, 1) = -lt;
        jacobians[4](0, 2) = -y * Real(0.5);
        jacobians[4](1, 0) = lt;
        jacobians[4](1, 2) = x * Real(0.5);

        jacobians[5](0, 1) = -lt;
        jacobians[5](0, 2) = -y * Real(0.5);
        jacobians[5](1, 0) = lt;
        jacobians[5](1, 2) = (x - Real(1)) * Real(0.5);

        jacobians[6](2, 0) = -Real(0.5);
        jacobians[6](2, 1) = -Real(0.5);
        jacobians[7](2, 0) = Real(0.5);
        jacobians[8](2, 1) = Real(0.5);
        return;
    }

    if (is_pyramid(element_type_)) {
        jacobians[0](0, 1) = -Real(0.25);
        jacobians[0](2, 0) = (Real(4) - Real(3) * y) / Real(8);
        jacobians[0](2, 1) = -Real(3) * x / Real(8);

        jacobians[1](1, 0) = Real(0.25);
        jacobians[1](2, 0) = Real(3) * y / Real(8);
        jacobians[1](2, 1) = (Real(3) * x + Real(4)) / Real(8);

        jacobians[2](0, 1) = -Real(0.25);
        jacobians[2](2, 0) = (-Real(3) * y - Real(4)) / Real(8);
        jacobians[2](2, 1) = -Real(3) * x / Real(8);

        jacobians[3](1, 0) = Real(0.25);
        jacobians[3](2, 0) = Real(3) * y / Real(8);
        jacobians[3](2, 1) = (Real(3) * x - Real(4)) / Real(8);

        jacobians[4](2, 0) = Real(3) * y / Real(4) - Real(0.5);
        jacobians[4](2, 1) = Real(3) * x / Real(4) - Real(0.5);

        jacobians[5](2, 0) = -Real(3) * y / Real(4) + Real(0.5);
        jacobians[5](2, 1) = -Real(3) * x / Real(4) - Real(0.5);

        jacobians[6](2, 0) = Real(3) * y / Real(4) + Real(0.5);
        jacobians[6](2, 1) = Real(3) * x / Real(4) + Real(0.5);

        jacobians[7](2, 0) = -Real(3) * y / Real(4) - Real(0.5);
        jacobians[7](2, 1) = -Real(3) * x / Real(4) + Real(0.5);
        return;
    }

    if (is_tetrahedron(element_type_)) {
        jacobians[0](0, 1) = -Real(1);
        jacobians[0](0, 2) = -Real(1);
        jacobians[0](1, 0) = Real(1);
        jacobians[0](2, 0) = Real(1);

        jacobians[1](0, 1) = -Real(1);
        jacobians[1](1, 0) = Real(1);

        jacobians[2](0, 1) = -Real(1);
        jacobians[2](1, 0) = Real(1);
        jacobians[2](1, 2) = Real(1);
        jacobians[2](2, 1) = -Real(1);

        jacobians[3](0, 2) = Real(1);
        jacobians[3](1, 2) = Real(1);
        jacobians[3](2, 0) = -Real(1);
        jacobians[3](2, 1) = -Real(1);

        jacobians[4](0, 2) = -Real(1);
        jacobians[4](2, 0) = Real(1);

        jacobians[5](1, 2) = -Real(1);
        jacobians[5](2, 1) = Real(1);
        return;
    }

    jacobians[0](0, 1) = -Real(0.125) * (Real(1) - z);
    jacobians[0](0, 2) = -Real(0.125) * (Real(1) - y);
    jacobians[1](1, 0) = Real(0.125) * (Real(1) - z);
    jacobians[1](1, 2) = -Real(0.125) * (Real(1) + x);
    jacobians[2](0, 1) = -Real(0.125) * (Real(1) - z);
    jacobians[2](0, 2) = Real(0.125) * (Real(1) + y);
    jacobians[3](1, 0) = Real(0.125) * (Real(1) - z);
    jacobians[3](1, 2) = Real(0.125) * (Real(1) - x);

    jacobians[4](0, 1) = -Real(0.125) * (Real(1) + z);
    jacobians[4](0, 2) = Real(0.125) * (Real(1) - y);
    jacobians[5](1, 0) = Real(0.125) * (Real(1) + z);
    jacobians[5](1, 2) = Real(0.125) * (Real(1) + x);
    jacobians[6](0, 1) = -Real(0.125) * (Real(1) + z);
    jacobians[6](0, 2) = -Real(0.125) * (Real(1) + y);
    jacobians[7](1, 0) = Real(0.125) * (Real(1) + z);
    jacobians[7](1, 2) = -Real(0.125) * (Real(1) - x);

    jacobians[8](2, 0) = -Real(0.125) * (Real(1) - y);
    jacobians[8](2, 1) = -Real(0.125) * (Real(1) - x);
    jacobians[9](2, 0) = Real(0.125) * (Real(1) - y);
    jacobians[9](2, 1) = -Real(0.125) * (Real(1) + x);
    jacobians[10](2, 0) = Real(0.125) * (Real(1) + y);
    jacobians[10](2, 1) = Real(0.125) * (Real(1) + x);
    jacobians[11](2, 0) = -Real(0.125) * (Real(1) + y);
    jacobians[11](2, 1) = Real(0.125) * (Real(1) - x);
}

void NedelecBasis::evaluate_curl(const math::Vector<Real, 3>& xi,
                                 std::vector<math::Vector<Real, 3>>& curl) const {
    if (use_transformed_direct_seed_) {
        const std::size_t n = size_;
        const std::size_t num_seed = (order_ <= 2) ? n : 0u;
        const std::size_t num_extra = transformed_monomial_candidates_.size();
        const std::size_t candidate_count = num_seed + num_extra;
        curl.assign(n, math::Vector<Real, 3>{});

        FE_CHECK_ARG(transformed_sparse_coeffs_.rows == candidate_count &&
                         transformed_sparse_coeffs_.cols == n &&
                         transformed_sparse_coeffs_.row_offsets.size() == candidate_count + 1u,
                     "NedelecBasis::evaluate_curl: transformed ND sparse coefficient size mismatch");

        auto add_candidate_curl = [&](std::size_t candidate, const Vec3& seed_curl) {
            const std::size_t row_begin = transformed_sparse_coeffs_.row_offsets[candidate];
            const std::size_t row_end = transformed_sparse_coeffs_.row_offsets[candidate + 1u];
            for (std::size_t entry = row_begin; entry < row_end; ++entry) {
                const std::size_t dof = transformed_sparse_coeffs_.dofs[entry];
                const Real c = transformed_sparse_coeffs_.coefficients[entry];
                curl[dof][0] += c * seed_curl[0];
                curl[dof][1] += c * seed_curl[1];
                curl[dof][2] += c * seed_curl[2];
            }
        };

        if (num_seed > 0) {
            auto& seed_curl = vector_basis_scratch().vector_values;
            eval_nd_seed_curl(element_type_, order_, xi, seed_curl);
            FE_CHECK_ARG(seed_curl.size() == n,
                         "NedelecBasis::evaluate_curl: ND seed curl size mismatch");

            for (std::size_t p = 0; p < n; ++p) {
                add_candidate_curl(p, seed_curl[p]);
            }
        }

        if (num_extra > 0) {
            auto& scratch = vector_basis_scratch();
            fill_power_tables(xi, transformed_power_limits_, scratch);
            const auto& px = scratch.px;
            const auto& py = scratch.py;
            const auto& pz = scratch.pz;

            std::size_t candidate = num_seed;
            for (const auto& mono : transformed_monomial_candidates_) {
                const Vec3 mono_curl =
                    eval_transformed_nd_monomial_curl(mono, px, py, pz);
                if (mono_curl[0] != Real(0) || mono_curl[1] != Real(0) ||
                    mono_curl[2] != Real(0)) {
                    add_candidate_curl(candidate, mono_curl);
                }
                ++candidate;
            }
        }
        return;
    }

    if (nodal_generated_) {
        evaluate_nodal_modal_vector_curl_with_limits(
            monomials_, modal_sparse_coeffs_, size_, xi, modal_power_limits_, curl);
        return;
    }

    if (dimension_ == 2) {
        // 2D curl stored in z-component
        if (is_triangle(element_type_)) {
            curl.resize(3);
            // All three simple triangle edge fields have constant curl = 2
            curl[0] = math::Vector<Real, 3>{Real(0), Real(0), Real(2)};
            curl[1] = math::Vector<Real, 3>{Real(0), Real(0), Real(2)};
            curl[2] = math::Vector<Real, 3>{Real(0), Real(0), Real(2)};
        } else {
            curl.resize(4);
            // For the Quad4-oriented edge basis in evaluate_vector_values,
            // all curls are constant and equal to 1/4 in the z-direction.
            curl[0] = math::Vector<Real, 3>{Real(0), Real(0), Real(0.25)};
            curl[1] = math::Vector<Real, 3>{Real(0), Real(0), Real(0.25)};
            curl[2] = math::Vector<Real, 3>{Real(0), Real(0), Real(0.25)};
            curl[3] = math::Vector<Real, 3>{Real(0), Real(0), Real(0.25)};
        }
        (void)xi;
        return;
    }

    if (is_wedge(element_type_)) {
        const Real x = xi[0];
        const Real y = xi[1];
        const Real z = xi[2];
        curl.resize(9);
        // Bottom edges (z=-1): (triangle Nedelec0) × (1 - z)/2
        curl[0] = math::Vector<Real, 3>{ Real(0.5) * x,
                                         Real(0.5) * (y - Real(1)),
                                         Real(1) - z };
        curl[1] = math::Vector<Real, 3>{ Real(0.5) * x,
                                         Real(0.5) * y,
                                         Real(1) - z };
        curl[2] = math::Vector<Real, 3>{ Real(0.5) * (x - Real(1)),
                                         Real(0.5) * y,
                                         Real(1) - z };

        // Top edges (z=+1): (triangle Nedelec0) × (1 + z)/2
        curl[3] = math::Vector<Real, 3>{ -Real(0.5) * x,
                                         Real(0.5) * (Real(1) - y),
                                         Real(1) + z };
        curl[4] = math::Vector<Real, 3>{ -Real(0.5) * x,
                                         -Real(0.5) * y,
                                         Real(1) + z };
        curl[5] = math::Vector<Real, 3>{ Real(0.5) * (Real(1) - x),
                                         -Real(0.5) * y,
                                         Real(1) + z };

        // Vertical edges
        curl[6] = math::Vector<Real, 3>{ -Real(0.5), Real(0.5), Real(0) };
        curl[7] = math::Vector<Real, 3>{ Real(0), -Real(0.5), Real(0) };
        curl[8] = math::Vector<Real, 3>{ Real(0.5), Real(0), Real(0) };
        return;
    }

    if (is_pyramid(element_type_)) {
        const Real x = xi[0];
        const Real y = xi[1];
        curl.resize(8);
        // From SymPy derivation:
        // curl φ0 = (-3*x/8,  3*y/8 - 1/2, 1/4)
        curl[0] = math::Vector<Real, 3>{
            -Real(3) * x / Real(8),
            Real(3) * y / Real(8) - Real(1) / Real(2),
            Real(1) / Real(4)
        };
        // curl φ1 = ( 3*x/8 + 1/2, -3*y/8, 1/4)
        curl[1] = math::Vector<Real, 3>{
            Real(3) * x / Real(8) + Real(1) / Real(2),
            -Real(3) * y / Real(8),
            Real(1) / Real(4)
        };
        // curl φ2 = (-3*x/8,  3*y/8 + 1/2, 1/4)
        curl[2] = math::Vector<Real, 3>{
            -Real(3) * x / Real(8),
            Real(3) * y / Real(8) + Real(1) / Real(2),
            Real(1) / Real(4)
        };
        // curl φ3 = ( 3*x/8 - 1/2, -3*y/8, 1/4)
        curl[3] = math::Vector<Real, 3>{
            Real(3) * x / Real(8) - Real(1) / Real(2),
            -Real(3) * y / Real(8),
            Real(1) / Real(4)
        };
        // curl φ4 = ( 3*x/4 - 1/2,  1/2 - 3*y/4, 0)
        curl[4] = math::Vector<Real, 3>{
            Real(3) * x / Real(4) - Real(1) / Real(2),
            Real(1) / Real(2) - Real(3) * y / Real(4),
            Real(0)
        };
        // curl φ5 = (-3*x/4 - 1/2,  3*y/4 - 1/2, 0)
        curl[5] = math::Vector<Real, 3>{
            -Real(3) * x / Real(4) - Real(1) / Real(2),
            Real(3) * y / Real(4) - Real(1) / Real(2),
            Real(0)
        };
        // curl φ6 = ( 3*x/4 + 1/2, -3*y/4 - 1/2, 0)
        curl[6] = math::Vector<Real, 3>{
            Real(3) * x / Real(4) + Real(1) / Real(2),
            -Real(3) * y / Real(4) - Real(1) / Real(2),
            Real(0)
        };
        // curl φ7 = ( 1/2 - 3*x/4,  3*y/4 + 1/2, 0)
        curl[7] = math::Vector<Real, 3>{
            Real(1) / Real(2) - Real(3) * x / Real(4),
            Real(3) * y / Real(4) + Real(1) / Real(2),
            Real(0)
        };
        return;
    }

    if (is_tetrahedron(element_type_)) {
        // Curls of the 6 tetra edge basis functions in evaluate_vector_values.
        curl.resize(6, math::Vector<Real, 3>{});
        curl[0] = math::Vector<Real, 3>{Real(0), Real(-2), Real(2)};
        curl[1] = math::Vector<Real, 3>{Real(0), Real(0), Real(2)};
        curl[2] = math::Vector<Real, 3>{Real(-2), Real(0), Real(2)};
        curl[3] = math::Vector<Real, 3>{Real(-2), Real(2), Real(0)};
        curl[4] = math::Vector<Real, 3>{Real(0), Real(-2), Real(0)};
        curl[5] = math::Vector<Real, 3>{Real(2), Real(0), Real(0)};
        return;
    }

    // Hexahedron: curls of the 12 edge basis functions.
    curl.resize(12, math::Vector<Real, 3>{});
    const Real x = xi[0];
    const Real y = xi[1];
    const Real z = xi[2];

    // Bottom face edges (z=-1)
    curl[0][1] = -Real(0.125) * (Real(1) - y);
    curl[0][2] =  Real(0.125) * (Real(1) - z);

    curl[1][0] =  Real(0.125) * (Real(1) + x);
    curl[1][2] =  Real(0.125) * (Real(1) - z);

    curl[2][1] =  Real(0.125) * (Real(1) + y);
    curl[2][2] =  Real(0.125) * (Real(1) - z);

    curl[3][0] = -Real(0.125) * (Real(1) - x);
    curl[3][2] =  Real(0.125) * (Real(1) - z);

    // Top face edges (z=+1)
    curl[4][1] =  Real(0.125) * (Real(1) - y);
    curl[4][2] =  Real(0.125) * (Real(1) + z);

    curl[5][0] = -Real(0.125) * (Real(1) + x);
    curl[5][2] =  Real(0.125) * (Real(1) + z);

    curl[6][1] = -Real(0.125) * (Real(1) + y);
    curl[6][2] =  Real(0.125) * (Real(1) + z);

    curl[7][0] =  Real(0.125) * (Real(1) - x);
    curl[7][2] =  Real(0.125) * (Real(1) + z);

    // Vertical edges
    curl[8][0] = -Real(0.125) * (Real(1) - x);
    curl[8][1] =  Real(0.125) * (Real(1) - y);

    curl[9][0] = -Real(0.125) * (Real(1) + x);
    curl[9][1] = -Real(0.125) * (Real(1) - y);

    curl[10][0] =  Real(0.125) * (Real(1) + x);
    curl[10][1] = -Real(0.125) * (Real(1) + y);

    curl[11][0] =  Real(0.125) * (Real(1) - x);
    curl[11][1] =  Real(0.125) * (Real(1) + y);
    return;
}

void NedelecBasis::evaluate_vector_at_quadrature_points_strided(
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
            "NedelecBasis");
        return;
    }

    if (use_transformed_direct_seed_) {
        const std::size_t num_qpts = points.size();
        const std::size_t n = size_;
        const std::size_t num_seed = (order_ <= 2) ? n : 0u;
        const std::size_t num_extra = transformed_monomial_candidates_.size();
        const std::size_t candidate_count = num_seed + num_extra;
        validate_vector_strided_outputs(num_qpts, output_stride, "NedelecBasis");
        FE_CHECK_ARG(transformed_sparse_coeffs_.rows == candidate_count &&
                         transformed_sparse_coeffs_.cols == n &&
                         transformed_sparse_coeffs_.row_offsets.size() == candidate_count + 1u,
                     "NedelecBasis strided transformed ND sparse coefficient size mismatch");

        auto& scratch = vector_basis_scratch();
        const bool need_values = values_out != nullptr;
        const bool need_jacobians = jacobians_out != nullptr;
        const bool need_curls = curls_out != nullptr;
        const bool need_divergence = divergence_out != nullptr;
        const bool need_derivative_tensor = need_jacobians || need_divergence;
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
                if (need_values) {
                    eval_nd_seed_values(element_type_, order_, points[q], scratch.api_values);
                    FE_CHECK_ARG(scratch.api_values.size() == n,
                                 "NedelecBasis strided ND seed value size mismatch");
                }
                if (need_derivative_tensor) {
                    FE_CHECK_ARG(transformed_seed_jacobian_evaluator_ != nullptr,
                                 "NedelecBasis strided transformed ND seed Jacobian evaluator is not configured");
                    transformed_seed_jacobian_evaluator_(points[q], scratch.api_jacobians);
                    FE_CHECK_ARG(scratch.api_jacobians.size() == n,
                                 "NedelecBasis strided ND seed Jacobian size mismatch");
                } else if (need_curls) {
                    eval_nd_seed_curl(element_type_, order_, points[q], scratch.api_curl);
                    FE_CHECK_ARG(scratch.api_curl.size() == n,
                                 "NedelecBasis strided ND seed curl size mismatch");
                }

                for (std::size_t seed = 0; seed < num_seed; ++seed) {
                    const Vec3 seed_value = need_values ? scratch.api_values[seed] : Vec3{};
                    const VectorJacobian seed_jacobian =
                        need_derivative_tensor ? scratch.api_jacobians[seed] : VectorJacobian{};
                    const Vec3 seed_curl =
                        need_derivative_tensor ? curl_from_jacobian(seed_jacobian)
                                               : need_curls ? scratch.api_curl[seed]
                                                            : Vec3{};
                    const Real seed_divergence =
                        need_derivative_tensor ? divergence_from_jacobian(seed_jacobian)
                                               : Real(0);

                    const std::size_t row_begin = transformed_sparse_coeffs_.row_offsets[seed];
                    const std::size_t row_end = transformed_sparse_coeffs_.row_offsets[seed + 1u];
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
                                       true,
                                       false,
                                       "NedelecBasis");
}

} // namespace basis
} // namespace FE
} // namespace svmp
