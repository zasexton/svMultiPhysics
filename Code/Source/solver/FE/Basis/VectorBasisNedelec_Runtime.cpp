/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "VectorBasis.h"
#include "Basis/BasisTraits.h"


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


void NedelecBasis::evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                          std::vector<math::Vector<Real, 3>>& values) const {
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
