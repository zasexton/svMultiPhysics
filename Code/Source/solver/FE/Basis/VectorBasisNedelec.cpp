/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "VectorBasis.h"
#include "Basis/BasisTraits.h"
#include "Basis/LagrangeBasis.h"
#include "Basis/NodeOrderingConventions.h"
#include "Elements/ReferenceElement.h"
#include "Quadrature/QuadratureFactory.h"
#include "Quadrature/ReferenceMonomialIntegrals.h"
#include "VectorBasisDirectSeeds.h"
#include "VectorBasisRtConstruction.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <utility>
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

using detail::vector_construction::build_nd_direct_transform;
using detail::vector_construction::eval_nd_seed_curl;
using detail::vector_construction::eval_nd_seed_values;
using detail::vector_construction::invert_dense_matrix;
using quadrature::reference_integrals::integral_monomial_1d;
using quadrature::reference_integrals::integral_pyramid_z;
using quadrature::reference_integrals::integral_tetra_monomial;
using quadrature::reference_integrals::integral_triangle_monomial;

// ----------------------------------------------------------------------------- //

NedelecBasis::NedelecBasis(ElementType type, int order)
    : element_type_(type), dimension_(0), order_(order) {
    if (order_ < 0) {
        throw BasisConfigurationException("NedelecBasis requires non-negative order",
                                          __FILE__, __LINE__, __func__);
    }

    ensure_supported_hybrid_vector_order(type, order_, "NedelecBasis");

    if (is_triangle(type) || is_quadrilateral(type)) {
        dimension_ = 2;
        if (is_triangle(type)) {
            // Simplex Nédélec (first kind) in 2D: [P_k]^2 ⊕ (-y,x) * \tilde P_k (homogeneous degree k)
            const std::size_t k = static_cast<std::size_t>(order_);
            size_ = (k + 1u) * (k + 3u);
        } else {
            const std::size_t k = static_cast<std::size_t>(order_);
            size_ = 2u * (k + 1u) * (k + 2u);
        }
    } else if (is_tetrahedron(type)) {
        dimension_ = 3;
        // Simplex Nédélec (first kind) in 3D:
        //   ND_k = [P_k]^3 ⊕ (x × \tilde P_k^3)
        // with dimension (k+1)(k+3)(k+4)/2.
        const std::size_t k = static_cast<std::size_t>(order_);
        size_ = (k + 1u) * (k + 3u) * (k + 4u) / 2u;
    } else if (is_hexahedron(type)) {
        dimension_ = 3;
        const std::size_t k = static_cast<std::size_t>(order_);
        size_ = 3u * (k + 1u) * (k + 2u) * (k + 2u);
    } else if (is_wedge(type)) {
        dimension_ = 3;
        size_ = nd_wedge_size(order_);
    } else if (is_pyramid(type)) {
        dimension_ = 3;
        size_ = nd_pyramid_size(order_);
    } else {
        throw BasisElementCompatibilityException("NedelecBasis supports triangles/quadrilaterals (2D) and "
                                                 "tetrahedra/hexahedra/wedges/pyramids (3D)",
                                                 __FILE__, __LINE__, __func__);
    }

    // Wedge/pyramid ND(1-2) uses the explicit seed formulas transformed into a
    // nodal basis with the actual edge/face/interior DOF functionals. For k>=3
    // we use the same DOF solve over an overcomplete polynomial candidate space.
    if (order_ >= 1 && (is_wedge(type) || is_pyramid(type))) {
        transformed_monomial_candidates_ = make_nd_extra_monomial_candidates(type, order_);
        transformed_power_limits_ = component_monomial_power_limits(transformed_monomial_candidates_);
        const std::vector<Real> coeffs =
            build_nd_direct_transform(type, order_, size_, transformed_monomial_candidates_);
        const std::size_t num_seed = (order_ <= 2) ? size_ : 0u;
        if (num_seed > 0) {
            if (is_wedge(type)) {
                transformed_seed_jacobian_evaluator_ =
                    order_ == 1
                        ? &detail::vector_direct::eval_wedge_nd1_jacobians
                        : &detail::vector_direct::eval_wedge_nd2_jacobians;
            } else {
                transformed_seed_jacobian_evaluator_ =
                    order_ == 1
                        ? &detail::vector_direct::eval_pyramid_nd1_jacobians
                        : &detail::vector_direct::eval_pyramid_nd2_jacobians;
            }
        }
        transformed_sparse_coeffs_ = build_sparse_modal_coefficients(
            coeffs,
            num_seed + transformed_monomial_candidates_.size(),
            size_);
        use_transformed_direct_seed_ = true;
        return;
    }

    if (order_ > 0 && (is_quadrilateral(type) || is_hexahedron(type) ||
                       is_triangle(type) || is_tetrahedron(type) ||
                       is_wedge(type) || is_pyramid(type))) {
        const std::size_t n = size_;

        monomials_.clear();
        monomials_.reserve(n);

        const int k = order_;
        auto push_single = [&](int component, int px, int py, int pz) {
            ModalPolynomial poly;
            poly.num_terms = 1;
            poly.terms[0] = ModalTerm{component, px, py, pz, Real(1)};
            append_unique_modal_polynomial(monomials_, poly);
        };

        if (dimension_ == 2) {
            if (is_triangle(type)) {
                // [P_k]^2
                for (int j = 0; j <= k; ++j) {
                    for (int i = 0; i <= k - j; ++i) {
                        push_single(0, i, j, 0);
                    }
                }
                for (int j = 0; j <= k; ++j) {
                    for (int i = 0; i <= k - j; ++i) {
                        push_single(1, i, j, 0);
                    }
                }
                // (-y,x) * \tilde P_k (homogeneous degree k): q_i = x^i y^{k-i}
                for (int i = 0; i <= k; ++i) {
                    ModalPolynomial poly;
                    poly.num_terms = 2;
                    poly.terms[0] = ModalTerm{0, i, k - i + 1, 0, Real(-1)}; // -y*q
                    poly.terms[1] = ModalTerm{1, i + 1, k - i, 0, Real(1)};  //  x*q
                    append_unique_modal_polynomial(monomials_, poly);
                }
            } else {
                // Tensor-product ND(k): Q_{k,k+1}×Q_{k+1,k}
                // x-component: i=0..k, j=0..k+1
                for (int j = 0; j <= k + 1; ++j) {
                    for (int i = 0; i <= k; ++i) {
                        push_single(0, i, j, 0);
                    }
                }
                // y-component: i=0..k+1, j=0..k
                for (int j = 0; j <= k; ++j) {
                    for (int i = 0; i <= k + 1; ++i) {
                        push_single(1, i, j, 0);
                    }
                }
            }
        } else {
            if (is_tetrahedron(type)) {
                // Simplex ND(k): [P_k]^3 ⊕ (x × \tilde P_k^3).
                // [P_k]^3
                for (int pz = 0; pz <= k; ++pz) {
                    for (int py = 0; py <= k - pz; ++py) {
                        for (int px = 0; px <= k - pz - py; ++px) {
                            push_single(0, px, py, pz);
                        }
                    }
                }
                for (int pz = 0; pz <= k; ++pz) {
                    for (int py = 0; py <= k - pz; ++py) {
                        for (int px = 0; px <= k - pz - py; ++px) {
                            push_single(1, px, py, pz);
                        }
                    }
                }
                for (int pz = 0; pz <= k; ++pz) {
                    for (int py = 0; py <= k - pz; ++py) {
                        for (int px = 0; px <= k - pz - py; ++px) {
                            push_single(2, px, py, pz);
                        }
                    }
                }

                // x × \tilde P_k^3 (homogeneous degree k): build a kernel-free spanning set.
                // Use q = (0,m,0) and q = (0,0,m) for all homogeneous monomials m of degree k,
                // and q = (m,0,0) for monomials with no x factor (px=0) only. This avoids the
                // kernel q = x * p (parallel-to-x vectors).
                //
                // For q=(0,m,0): x×q = (-z*m, 0, x*m)
                for (int pz = 0; pz <= k; ++pz) {
                    for (int py = 0; py <= k - pz; ++py) {
                        const int px = k - pz - py;
                        ModalPolynomial poly;
                        poly.num_terms = 2;
                        poly.terms[0] = ModalTerm{0, px, py, pz + 1, Real(-1)};
                        poly.terms[1] = ModalTerm{2, px + 1, py, pz, Real(1)};
                        append_unique_modal_polynomial(monomials_, poly);
                    }
                }

                // For q=(0,0,m): x×q = (y*m, -x*m, 0)
                for (int pz = 0; pz <= k; ++pz) {
                    for (int py = 0; py <= k - pz; ++py) {
                        const int px = k - pz - py;
                        ModalPolynomial poly;
                        poly.num_terms = 2;
                        poly.terms[0] = ModalTerm{0, px, py + 1, pz, Real(1)};
                        poly.terms[1] = ModalTerm{1, px + 1, py, pz, Real(-1)};
                        append_unique_modal_polynomial(monomials_, poly);
                    }
                }

                // For q=(m,0,0) with px=0: x×q = (0, z*m, -y*m), m = y^py z^pz, py+pz=k.
                for (int py = 0; py <= k; ++py) {
                    const int pz = k - py;
                    ModalPolynomial poly;
                    poly.num_terms = 2;
                    poly.terms[0] = ModalTerm{1, 0, py, pz + 1, Real(1)};
                    poly.terms[1] = ModalTerm{2, 0, py + 1, pz, Real(-1)};
                    append_unique_modal_polynomial(monomials_, poly);
                }
            } else if (is_hexahedron(type)) {
                // Tensor-product ND(k) on hex.
                // x-component: i=0..k, j=0..k+1, l=0..k+1
                for (int l = 0; l <= k + 1; ++l) {
                    for (int j = 0; j <= k + 1; ++j) {
                        for (int i = 0; i <= k; ++i) {
                            push_single(0, i, j, l);
                        }
                    }
                }
                // y-component: i=0..k+1, j=0..k, l=0..k+1
                for (int l = 0; l <= k + 1; ++l) {
                    for (int j = 0; j <= k; ++j) {
                        for (int i = 0; i <= k + 1; ++i) {
                            push_single(1, i, j, l);
                        }
                    }
                }
                // z-component: i=0..k+1, j=0..k+1, l=0..k
                for (int l = 0; l <= k; ++l) {
                    for (int j = 0; j <= k + 1; ++j) {
                        for (int i = 0; i <= k + 1; ++i) {
                            push_single(2, i, j, l);
                        }
                    }
                }
            } else if (is_wedge(type)) {
                // ================================================================
                // Wedge Nedelec(k): prismatic H(curl) element - CORRECTED CONSTRUCTION
                // ================================================================
                //
                // Reference: Zaglmayr (2006), Demkowicz et al.
                //
                // The Nedelec(k) space on a prism uses a hierarchical construction:
                // 1. Edge basis functions: one per edge per Legendre mode up to k
                // 2. Face tangential functions: for k>=1
                // 3. Interior functions: for k>=2
                //
                // DOF structure:
                // - 9 edges x (k+1) edge DOFs = 9(k+1)
                // - 2 tri faces x k(k+1) tangential DOFs + 3 quad faces x 2k(k+1) = 2k(k+1) + 6k(k+1) = 8k(k+1) for k>=1
                // - Interior: 3*k(k-1)(k+1)/2 for k>=2
                //
                // We use a hierarchical construction that matches this DOF count.

                // --- Phase 1: Edge-associated basis functions ---
                // Each edge has (k+1) basis functions

                // Bottom triangle edges (3 edges, each with k+1 functions)
                // Edge 0: (0,0,-1) to (1,0,-1)
                for (int a = 0; a <= k; ++a) {
                    push_single(0, a, 0, 0); // x^a at z=-1 level (approx)
                }
                // Edge 1: (1,0,-1) to (0,1,-1)
                for (int a = 0; a <= k; ++a) {
                    // Use y-directed function
                    push_single(1, 0, a, 0);
                }
                // Edge 2: (0,1,-1) to (0,0,-1)
                for (int a = 0; a <= k; ++a) {
                    push_single(1, a, 0, 0);
                }

                // Top triangle edges (3 edges, each with k+1 functions)
                // Similar to bottom but at z=+1 level (use z^1 factor)
                for (int a = 0; a <= k; ++a) {
                    push_single(0, a, 0, 1);
                }
                for (int a = 0; a <= k; ++a) {
                    push_single(1, 0, a, 1);
                }
                for (int a = 0; a <= k; ++a) {
                    push_single(1, a, 0, 1);
                }

                // Vertical edges (3 edges, each with k+1 functions)
                // These are z-directed
                for (int a = 0; a <= k; ++a) {
                    push_single(2, 0, 0, a);
                }
                for (int a = 0; a <= k; ++a) {
                    push_single(2, 1, 0, a);
                }
                for (int a = 0; a <= k; ++a) {
                    push_single(2, 0, 1, a);
                }

                // --- Phase 2: Face tangential functions (for k >= 1) ---
                if (k >= 1) {
                    // Triangular faces: 2 * k(k+1) DOFs
                    // Use tangential polynomials on the triangle face
                    for (int j = 0; j <= k - 1; ++j) {
                        for (int i = 0; i <= k - 1 - j; ++i) {
                            // Bottom tri face - two tangent directions
                            push_single(0, i, j, 0);
                            push_single(1, i, j, 0);
                        }
                    }
                    for (int j = 0; j <= k - 1; ++j) {
                        for (int i = 0; i <= k - 1 - j; ++i) {
                            // Top tri face
                            push_single(0, i, j, 1);
                            push_single(1, i, j, 1);
                        }
                    }

                    // Quad faces: 3 * 2k(k+1) DOFs
                    // Each quad face has two tangent directions
                    for (int l = 0; l <= k - 1; ++l) {
                        for (int i = 0; i <= k; ++i) {
                            // Face y=0: tangents are (1,0,0) and (0,0,1)
                            push_single(0, i, 0, l);
                            push_single(2, i, 0, l);
                        }
                    }
                    for (int l = 0; l <= k - 1; ++l) {
                        for (int j = 0; j <= k; ++j) {
                            // Face x=0: tangents are (0,1,0) and (0,0,1)
                            push_single(1, 0, j, l);
                            push_single(2, 0, j, l);
                        }
                    }
                    for (int l = 0; l <= k - 1; ++l) {
                        for (int s = 0; s <= k; ++s) {
                            // Face x+y=1: tangents are (-1,1,0) and (0,0,1)
                            // Use combined functions
                            ModalPolynomial poly;
                            poly.num_terms = 2;
                            poly.terms[0] = ModalTerm{0, s, 0, l, Real(-1)};
                            poly.terms[1] = ModalTerm{1, s, 0, l, Real(1)};
                            append_unique_modal_polynomial(monomials_, poly);
                            push_single(2, s, 0, l);
                        }
                    }
                }

                // --- Phase 3: Interior functions (for k >= 2) ---
                if (k >= 2) {
                    // Interior DOFs: 3 * k(k-1)(k+1)/2
                    // Test against P_{k-2}(tri) x P_{k-1}(z) for each component
                    for (int l = 0; l <= k - 1; ++l) {
                        for (int j = 0; j <= k - 2; ++j) {
                            for (int i = 0; i <= k - 2 - j; ++i) {
                                push_single(0, i, j, l);
                                push_single(1, i, j, l);
                                push_single(2, i, j, l);
                            }
                        }
                    }
                }
            } else if (is_pyramid(type)) {
                // ================================================================
                // Pyramid Nedelec(k): H(curl) element on pyramid - CORRECTED CONSTRUCTION
                // ================================================================
                //
                // Reference: Nigam-Phillips (2012), Falk-Gatto-Monk (2011)
                //
                // The pyramid has 8 edges, 5 faces (1 quad + 4 triangles).
                //
                // DOF structure:
                // - 8 edges x (k+1) edge DOFs = 8(k+1)
                // - 1 quad face x 2k(k+1) + 4 tri faces x k(k+1) = 6k(k+1) for k>=1
                // - Interior: pyramid-specific for k>=2
                //
                // We use a hierarchical construction.

                // --- Phase 1: Edge-associated basis functions ---
                // 4 base quad edges + 4 lateral edges to apex

                // Base quad edges (4 edges)
                for (int a = 0; a <= k; ++a) {
                    push_single(0, a, 0, 0); // edge along x
                }
                for (int a = 0; a <= k; ++a) {
                    push_single(1, 0, a, 0); // edge along y
                }
                for (int a = 0; a <= k; ++a) {
                    push_single(0, a, 1, 0); // edge along x at y=1
                }
                for (int a = 0; a <= k; ++a) {
                    push_single(1, 1, a, 0); // edge along y at x=1
                }

                // Lateral edges to apex (4 edges)
                // These go from base corners toward apex
                for (int a = 0; a <= k; ++a) {
                    push_single(2, 0, 0, a);
                }
                for (int a = 0; a <= k; ++a) {
                    push_single(2, 1, 0, a);
                }
                for (int a = 0; a <= k; ++a) {
                    push_single(2, 0, 1, a);
                }
                for (int a = 0; a <= k; ++a) {
                    push_single(2, 1, 1, a);
                }

                // --- Phase 2: Face tangential functions (for k >= 1) ---
                if (k >= 1) {
                    // Quad face (base): 2k(k+1) DOFs - two tangent directions
                    for (int j = 0; j <= k - 1; ++j) {
                        for (int i = 0; i <= k; ++i) {
                            push_single(0, i, j, 0);
                        }
                    }
                    for (int i = 0; i <= k - 1; ++i) {
                        for (int j = 0; j <= k; ++j) {
                            push_single(1, i, j, 0);
                        }
                    }

                    // 4 triangular faces: k(k+1) DOFs each
                    // Use tangential components localized to each face
                    for (int a = 0; a <= k - 1; ++a) {
                        for (int b = 0; b <= k - 1 - a; ++b) {
                            // Face 0 (x=-1+z side)
                            push_single(1, 0, a, b);
                            push_single(2, 0, a, b);
                        }
                    }
                    for (int a = 0; a <= k - 1; ++a) {
                        for (int b = 0; b <= k - 1 - a; ++b) {
                            // Face 1 (x=1-z side)
                            push_single(1, 1, a, b);
                            push_single(2, 1, a, b);
                        }
                    }
                    for (int a = 0; a <= k - 1; ++a) {
                        for (int b = 0; b <= k - 1 - a; ++b) {
                            // Face 2 (y=-1+z side)
                            push_single(0, a, 0, b);
                            push_single(2, a, 0, b);
                        }
                    }
                    for (int a = 0; a <= k - 1; ++a) {
                        for (int b = 0; b <= k - 1 - a; ++b) {
                            // Face 3 (y=1-z side)
                            push_single(0, a, 1, b);
                            push_single(2, a, 1, b);
                        }
                    }
                }

                // --- Phase 3: Interior functions (for k >= 2) ---
                if (k >= 2) {
                    // Interior DOFs: pyramid-specific
                    for (int l = 0; l <= k - 2; ++l) {
                        for (int j = 0; j <= k - 2; ++j) {
                            for (int i = 0; i <= k - 2; ++i) {
                                push_single(0, i, j, l);
                                push_single(1, i, j, l);
                                push_single(2, i, j, l);
                            }
                        }
                    }
                }
            }
        }
        // Wedge and pyramid candidate sets can be oversized. Keep the first
        // deterministic, duplicate-pruned subset and fail below if it is short.
        if (monomials_.size() > n) {
            monomials_.erase(monomials_.begin() + static_cast<std::ptrdiff_t>(n),
                             monomials_.end());
        }

        // Verify modal basis size matches expected DOF count
        FE_CHECK_ARG(monomials_.size() == n,
                     "NedelecBasis: modal basis size mismatch (expected " +
                     std::to_string(n) + ", got " + std::to_string(monomials_.size()) + ")");

        modal_power_limits_ = modal_power_limits(monomials_);
        const int max_px = modal_power_limits_[0];
        const int max_py = modal_power_limits_[1];
        const int max_pz = modal_power_limits_[2];
        std::vector<Real> power_x;
        std::vector<Real> power_y;
        std::vector<Real> power_z;

        std::vector<Real> A(n * n, Real(0));
        const elements::ReferenceElement ref = elements::ReferenceElement::create(type);
        std::size_t row = 0;

        // Edge tangential moments: ∫_e (v·t) * l_i(s) ds, i=0..k.
        const LagrangeBasis edge_basis(ElementType::Line2, k);
        const auto edge_quad = quadrature::QuadratureFactory::create(
            ElementType::Line2, 2 * k + 2, QuadratureType::GaussLegendre);
        std::vector<Real> edge_values;

        for (std::size_t e = 0; e < ref.num_edges(); ++e) {
            const auto& en = ref.edge_nodes(e);
            FE_CHECK_ARG(en.size() == 2u, "Nedelec: expected 2 vertices per edge");
            const Vec3 p0 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(en[0]));
            const Vec3 p1 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(en[1]));
            const Vec3 t = normalize3(p1 - p0);
            const Real J = math::norm(p1 - p0) * Real(0.5);

            for (int a = 0; a <= k; ++a) {
                FE_CHECK_ARG(row < n, "Nedelec: row overflow in edge moments");
                for (std::size_t q = 0; q < edge_quad->num_points(); ++q) {
                    const Real s = edge_quad->point(q)[0];
                    const Real wq = edge_quad->weight(q);
                    edge_values.clear();
                    edge_basis.evaluate_values(Vec3{s, Real(0), Real(0)}, edge_values);
                    FE_CHECK_ARG(edge_values.size() == static_cast<std::size_t>(k + 1), "Nedelec: edge basis size mismatch");
                    const Vec3 xi = lerp(p0, p1, s);

                    fill_powers(xi[0], max_px, power_x);
                    fill_powers(xi[1], max_py, power_y);
                    fill_powers(xi[2], max_pz, power_z);

                    const Real wt = wq * J * edge_values[static_cast<std::size_t>(a)];
                    for (std::size_t p = 0; p < n; ++p) {
                        const auto& poly = monomials_[p];
                        Real dot = Real(0);
                        for (int tt = 0; tt < poly.num_terms; ++tt) {
                            const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                            const Real mv = power_x[static_cast<std::size_t>(mono.px)] *
                                            power_y[static_cast<std::size_t>(mono.py)] *
                                            power_z[static_cast<std::size_t>(mono.pz)];
                            dot += mono.coefficient * t[static_cast<std::size_t>(mono.component)] * mv;
                        }
                        A[row * n + p] += wt * dot;
                    }
                }
                ++row;
            }
        }

        if (dimension_ == 2) {
            if (is_triangle(type)) {
                // Interior moments: component-wise moments against P_{k-1} monomials on the simplex.
                for (int c = 0; c < 2; ++c) {
                    for (int j = 0; j <= k - 1; ++j) {
                        for (int i = 0; i <= k - 1 - j; ++i) {
                            FE_CHECK_ARG(row < n, "ND triangle: row overflow in interior moments");
                            for (std::size_t p = 0; p < n; ++p) {
                                const auto& poly = monomials_[p];
                                Real acc = Real(0);
                                for (int tt = 0; tt < poly.num_terms; ++tt) {
                                    const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                    if (mono.component != c) {
                                        continue;
                                    }
                                    acc += mono.coefficient *
                                           integral_triangle_monomial(mono.px + i, mono.py + j);
                                }
                                A[row * n + p] = acc;
                            }
                            ++row;
                        }
                    }
                }
            } else {
                // Interior moments against Q_{k,k-1}×Q_{k-1,k} monomials.
                // x-component: i=0..k, j=0..k-1
                for (int j = 0; j <= k - 1; ++j) {
                    for (int i = 0; i <= k; ++i) {
                        FE_CHECK_ARG(row < n, "ND quad: row overflow in interior x-moments");
                        for (std::size_t p = 0; p < n; ++p) {
                            const auto& poly = monomials_[p];
                            Real acc = Real(0);
                            for (int tt = 0; tt < poly.num_terms; ++tt) {
                                const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                if (mono.component != 0) {
                                    continue;
                                }
                                const int sx = mono.px + i;
                                const int sy = mono.py + j;
                                acc += mono.coefficient *
                                       integral_monomial_1d(sx) * integral_monomial_1d(sy);
                            }
                            A[row * n + p] = acc;
                        }
                        ++row;
                    }
                }
                // y-component: i=0..k-1, j=0..k
                for (int j = 0; j <= k; ++j) {
                    for (int i = 0; i <= k - 1; ++i) {
                        FE_CHECK_ARG(row < n, "ND quad: row overflow in interior y-moments");
                        for (std::size_t p = 0; p < n; ++p) {
                            const auto& poly = monomials_[p];
                            Real acc = Real(0);
                            for (int tt = 0; tt < poly.num_terms; ++tt) {
                                const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                if (mono.component != 1) {
                                    continue;
                                }
                                const int sx = mono.px + i;
                                const int sy = mono.py + j;
                                acc += mono.coefficient *
                                       integral_monomial_1d(sx) * integral_monomial_1d(sy);
                            }
                            A[row * n + p] = acc;
                        }
                        ++row;
                    }
                }
            }
        } else {
            if (is_tetrahedron(type)) {
                // Face tangential moments on simplex faces:
                //   u-directed: ∫ (v·t_u) * P_{k-1}(u,v) dS
                //   v-directed: ∫ (v·t_v) * P_{k-1}(u,v) dS
                const auto face_quad = quadrature::QuadratureFactory::create(
                    ElementType::Triangle3, 2 * k + 2, QuadratureType::GaussLegendre);
                const LagrangeBasis face_basis(ElementType::Triangle3, k - 1);
                const std::size_t n_face = face_basis.size();
                std::vector<Real> face_values;
                std::vector<Real> mono_dot_u(n, Real(0));
                std::vector<Real> mono_dot_v(n, Real(0));

                for (std::size_t f = 0; f < ref.num_faces(); ++f) {
                    const auto& fn = ref.face_nodes(f);
                    FE_CHECK_ARG(fn.size() == 3u, "ND tetra: expected tri face with 3 vertices");
                    const Vec3 v0 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[0]));
                    const Vec3 v1 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[1]));
                    const Vec3 v2 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[2]));
                    const Vec3 tu = v1 - v0; // covariant tangents (not normalized)
                    const Vec3 tv = v2 - v0;
                    const Real scale = cross3(tu, tv).norm();

                    FE_CHECK_ARG(row + 2u * n_face <= n, "ND tetra: row overflow in face moments");
                    const std::size_t row_u = row;
                    const std::size_t row_v = row + n_face;

                    for (std::size_t q = 0; q < face_quad->num_points(); ++q) {
                        const auto uv = face_quad->point(q);
                        const Real u = uv[0];
                        const Real v = uv[1];
                        const Real wq = face_quad->weight(q);

                        face_values.clear();
                        face_basis.evaluate_values(Vec3{u, v, Real(0)}, face_values);
                        FE_CHECK_ARG(face_values.size() == n_face, "ND tetra: face basis size mismatch");

                        const Vec3 xi = v0 + tu * u + tv * v;
                        fill_powers(xi[0], max_px, power_x);
                        fill_powers(xi[1], max_py, power_y);
                        fill_powers(xi[2], max_pz, power_z);

                        for (std::size_t p = 0; p < n; ++p) {
                            const auto& poly = monomials_[p];
                            Real du = Real(0);
                            Real dv = Real(0);
                            for (int tt = 0; tt < poly.num_terms; ++tt) {
                                const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                const Real mv =
                                    power_x[static_cast<std::size_t>(mono.px)] *
                                    power_y[static_cast<std::size_t>(mono.py)] *
                                    power_z[static_cast<std::size_t>(mono.pz)];
                                du += mono.coefficient * tu[static_cast<std::size_t>(mono.component)] * mv;
                                dv += mono.coefficient * tv[static_cast<std::size_t>(mono.component)] * mv;
                            }
                            mono_dot_u[p] = du;
                            mono_dot_v[p] = dv;
                        }

                        const Real wt = wq * scale;
                        for (std::size_t a = 0; a < n_face; ++a) {
                            const Real wa = wt * face_values[a];
                            if (wa == Real(0)) {
                                continue;
                            }
                            const std::size_t ru = row_u + a;
                            const std::size_t rv = row_v + a;
                            for (std::size_t p = 0; p < n; ++p) {
                                A[ru * n + p] += wa * mono_dot_u[p];
                                A[rv * n + p] += wa * mono_dot_v[p];
                            }
                        }
                    }

                    row += 2u * n_face;
                }

                // Interior moments: component-wise moments against P_{k-2} monomials on the simplex.
                if (k >= 2) {
                    for (int c = 0; c < 3; ++c) {
                        for (int pz = 0; pz <= k - 2; ++pz) {
                            for (int py = 0; py <= (k - 2) - pz; ++py) {
                                for (int px = 0; px <= (k - 2) - pz - py; ++px) {
                                    FE_CHECK_ARG(row < n, "ND tetra: row overflow in interior moments");
                                    for (std::size_t p = 0; p < n; ++p) {
                                        const auto& poly = monomials_[p];
                                        Real acc = Real(0);
                                        for (int tt = 0; tt < poly.num_terms; ++tt) {
                                            const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                            if (mono.component != c) {
                                                continue;
                                            }
                                            acc += mono.coefficient *
                                                   integral_tetra_monomial(mono.px + px,
                                                                         mono.py + py,
                                                                         mono.pz + pz);
                                        }
                                        A[row * n + p] = acc;
                                    }
                                    ++row;
                                }
                            }
                        }
                    }
                }
            } else if (is_hexahedron(type)) {
                // Face tangential moments:
                //   u-directed: ∫ (v·t_u) * Q_{k-1,k}(u,w) dS
                //   w-directed: ∫ (v·t_w) * Q_{k,k-1}(u,w) dS
                const auto face_quad = quadrature::QuadratureFactory::create(
                    ElementType::Quad4, 2 * k + 2, QuadratureType::GaussLegendre);

                const LagrangeBasis u_low(ElementType::Line2, k - 1);
                const LagrangeBasis u_full(ElementType::Line2, k);
                const LagrangeBasis w_low(ElementType::Line2, k - 1);
                const LagrangeBasis w_full(ElementType::Line2, k);
                std::vector<Real> u_low_values;
                std::vector<Real> u_full_values;
                std::vector<Real> w_low_values;
                std::vector<Real> w_full_values;
                std::vector<Real> mono_dot_u(n, Real(0));
                std::vector<Real> mono_dot_w(n, Real(0));

                for (std::size_t f = 0; f < ref.num_faces(); ++f) {
                    const auto& fn = ref.face_nodes(f);
                    FE_CHECK_ARG(fn.size() == 4u, "ND hex: expected quad face with 4 vertices");
                    const std::array<Vec3, 4> fv{
                        ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[0])),
                        ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[1])),
                        ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[2])),
                        ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[3]))
                    };
                    const Vec3 tu = normalize3(fv[1] - fv[0]);
                    const Vec3 tw = normalize3(fv[3] - fv[0]);

                    const std::size_t n_u = static_cast<std::size_t>(k) * static_cast<std::size_t>(k + 1);
                    const std::size_t n_w = static_cast<std::size_t>(k + 1) * static_cast<std::size_t>(k);
                    FE_CHECK_ARG(row + n_u + n_w <= n, "ND hex: row overflow in face moments");

                    for (std::size_t q = 0; q < face_quad->num_points(); ++q) {
                        const auto uv = face_quad->point(q);
                        const Real u = uv[0];
                        const Real w = uv[1];
                        const Real wq = face_quad->weight(q);

                        u_low_values.clear();
                        u_full_values.clear();
                        w_low_values.clear();
                        w_full_values.clear();
                        u_low.evaluate_values(Vec3{u, Real(0), Real(0)}, u_low_values);
                        u_full.evaluate_values(Vec3{u, Real(0), Real(0)}, u_full_values);
                        w_low.evaluate_values(Vec3{w, Real(0), Real(0)}, w_low_values);
                        w_full.evaluate_values(Vec3{w, Real(0), Real(0)}, w_full_values);
                        FE_CHECK_ARG(u_low_values.size() == static_cast<std::size_t>(k), "ND hex: u_low size mismatch");
                        FE_CHECK_ARG(u_full_values.size() == static_cast<std::size_t>(k + 1), "ND hex: u_full size mismatch");
                        FE_CHECK_ARG(w_low_values.size() == static_cast<std::size_t>(k), "ND hex: w_low size mismatch");
                        FE_CHECK_ARG(w_full_values.size() == static_cast<std::size_t>(k + 1), "ND hex: w_full size mismatch");

                        const Vec3 xi = bilinear(fv, u, w);
                        const Vec3 dxdu = bilinear_du(fv, u, w);
                        const Vec3 dxdw = bilinear_dw(fv, u, w);
                        const Real scale = cross3(dxdu, dxdw).norm();
                        const Real wt = wq * scale;

                        fill_powers(xi[0], max_px, power_x);
                        fill_powers(xi[1], max_py, power_y);
                        fill_powers(xi[2], max_pz, power_z);

                        for (std::size_t p = 0; p < n; ++p) {
                            const auto& poly = monomials_[p];
                            Real du = Real(0);
                            Real dwv = Real(0);
                            for (int tt = 0; tt < poly.num_terms; ++tt) {
                                const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                const Real mv = power_x[static_cast<std::size_t>(mono.px)] *
                                                power_y[static_cast<std::size_t>(mono.py)] *
                                                power_z[static_cast<std::size_t>(mono.pz)];
                                du += mono.coefficient * tu[static_cast<std::size_t>(mono.component)] * mv;
                                dwv += mono.coefficient * tw[static_cast<std::size_t>(mono.component)] * mv;
                            }
                            mono_dot_u[p] = du;
                            mono_dot_w[p] = dwv;
                        }

                        // u-directed moments: (j_w outer, i_u inner)
                        for (int jw = 0; jw <= k; ++jw) {
                            for (int iu = 0; iu <= k - 1; ++iu) {
                                const std::size_t a = static_cast<std::size_t>(jw) * static_cast<std::size_t>(k) + static_cast<std::size_t>(iu);
                                const Real basis_val = u_low_values[static_cast<std::size_t>(iu)] * w_full_values[static_cast<std::size_t>(jw)];
                                const Real wa = wt * basis_val;
                                const std::size_t r = row + a;
                                for (std::size_t p = 0; p < n; ++p) {
                                    A[r * n + p] += wa * mono_dot_u[p];
                                }
                            }
                        }

                        // w-directed moments start after u-block
                        const std::size_t row_w = row + n_u;
                        for (int jw = 0; jw <= k - 1; ++jw) {
                            for (int iu = 0; iu <= k; ++iu) {
                                const std::size_t a = static_cast<std::size_t>(jw) * static_cast<std::size_t>(k + 1) + static_cast<std::size_t>(iu);
                                const Real basis_val = u_full_values[static_cast<std::size_t>(iu)] * w_low_values[static_cast<std::size_t>(jw)];
                                const Real wa = wt * basis_val;
                                const std::size_t r = row_w + a;
                                for (std::size_t p = 0; p < n; ++p) {
                                    A[r * n + p] += wa * mono_dot_w[p];
                                }
                            }
                        }
                    }

                    row += n_u + n_w;
                }

                // Interior moments: component-wise moments against Q_{k-1,k-1,k} monomials.
                for (int c = 0; c < 3; ++c) {
                    for (int l = 0; l <= k; ++l) {
                        for (int j = 0; j <= k - 1; ++j) {
                            for (int i = 0; i <= k - 1; ++i) {
                                FE_CHECK_ARG(row < n, "ND hex: row overflow in interior moments");
                                for (std::size_t p = 0; p < n; ++p) {
                                    const auto& poly = monomials_[p];
                                    Real acc = Real(0);
                                    for (int tt = 0; tt < poly.num_terms; ++tt) {
                                        const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                        if (mono.component != c) {
                                            continue;
                                        }
                                        const int sx = mono.px + i;
                                        const int sy = mono.py + j;
                                        const int sz = mono.pz + l;
                                        acc += mono.coefficient *
                                               integral_monomial_1d(sx) *
                                               integral_monomial_1d(sy) *
                                               integral_monomial_1d(sz);
                                    }
                                    A[row * n + p] = acc;
                                }
                                ++row;
                            }
                        }
                    }
                }
            } else if (is_wedge(type) || is_pyramid(type)) {
                // Mixed face types for wedge/pyramid: handle triangular and quad faces
                const std::size_t modal_count = monomials_.size();

                // Face tangential moments on mixed faces
                const auto tri_quad = quadrature::QuadratureFactory::create(
                    ElementType::Triangle3, 2 * k + 2, QuadratureType::GaussLegendre);
                const auto quad_quad = quadrature::QuadratureFactory::create(
                    ElementType::Quad4, 2 * k + 2, QuadratureType::GaussLegendre);
                const LagrangeBasis tri_face_basis(ElementType::Triangle3, k - 1);
                const LagrangeBasis quad_u_basis(ElementType::Line2, k - 1);
                const LagrangeBasis quad_w_basis(ElementType::Line2, k - 1);
                std::vector<Real> face_values;
                std::vector<Real> quad_u_values;
                std::vector<Real> quad_w_values;
                std::vector<Real> mono_u(modal_count, Real(0));
                std::vector<Real> mono_v(modal_count, Real(0));
                std::vector<Real> mono_w(modal_count, Real(0));

                for (std::size_t f = 0; f < ref.num_faces(); ++f) {
                    const auto& fn = ref.face_nodes(f);
                    const bool is_tri = (fn.size() == 3u);

                    if (is_tri) {
                        // Triangular face tangential moments
                        const Vec3 v0 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[0]));
                        const Vec3 v1 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[1]));
                        const Vec3 v2 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[2]));
                        const Vec3 tu = v1 - v0;
                        const Vec3 tv = v2 - v0;
                        const Real scale = cross3(tu, tv).norm();

                        const std::size_t n_face = tri_face_basis.size();
                        if (row + 2u * n_face > n) break;

                        for (std::size_t q = 0; q < tri_quad->num_points(); ++q) {
                            const auto uv = tri_quad->point(q);
                            const Real u = uv[0];
                            const Real v = uv[1];
                            const Real wq = tri_quad->weight(q);

                            face_values.clear();
                            tri_face_basis.evaluate_values(Vec3{u, v, Real(0)}, face_values);

                            const Vec3 xi_pt = v0 + tu * u + tv * v;
                            fill_powers(xi_pt[0], max_px, power_x);
                            fill_powers(xi_pt[1], max_py, power_y);
                            fill_powers(xi_pt[2], max_pz, power_z);

                            for (std::size_t p = 0; p < modal_count; ++p) {
                                const auto& poly = monomials_[p];
                                Real du = Real(0), dv = Real(0);
                                for (int tt = 0; tt < poly.num_terms; ++tt) {
                                    const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                    const Real mv = power_x[static_cast<std::size_t>(mono.px)] *
                                                    power_y[static_cast<std::size_t>(mono.py)] *
                                                    power_z[static_cast<std::size_t>(mono.pz)];
                                    du += mono.coefficient * tu[static_cast<std::size_t>(mono.component)] * mv;
                                    dv += mono.coefficient * tv[static_cast<std::size_t>(mono.component)] * mv;
                                }
                                mono_u[p] = du;
                                mono_v[p] = dv;
                            }

                            const Real wt = wq * scale;
                            for (std::size_t a = 0; a < n_face; ++a) {
                                const Real wa = wt * face_values[a];
                                if (wa == Real(0)) continue;
                                for (std::size_t p = 0; p < modal_count; ++p) {
                                    A[(row + a) * n + p] += wa * mono_u[p];
                                    A[(row + n_face + a) * n + p] += wa * mono_v[p];
                                }
                            }
                        }
                        row += 2u * n_face;
                    } else {
                        // Quad face tangential moments
                        const std::array<Vec3, 4> fv{
                            ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[0])),
                            ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[1])),
                            ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[2])),
                            ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[3]))
                        };
                        const Vec3 tu = normalize3(fv[1] - fv[0]);
                        const Vec3 tw = normalize3(fv[3] - fv[0]);

                        const std::size_t n_u = static_cast<std::size_t>(k) * static_cast<std::size_t>(k + 1);
                        const std::size_t n_w = static_cast<std::size_t>(k + 1) * static_cast<std::size_t>(k);
                        if (row + n_u + n_w > n) break;

                        for (std::size_t q = 0; q < quad_quad->num_points(); ++q) {
                            const auto uv = quad_quad->point(q);
                            const Real u = uv[0];
                            const Real w = uv[1];
                            const Real wq = quad_quad->weight(q);

                            const Vec3 xi_pt = bilinear(fv, u, w);
                            const Vec3 dxdu = bilinear_du(fv, u, w);
                            const Vec3 dxdw = bilinear_dw(fv, u, w);
                            const Real scale = cross3(dxdu, dxdw).norm();

                            fill_powers(xi_pt[0], max_px, power_x);
                            fill_powers(xi_pt[1], max_py, power_y);
                            fill_powers(xi_pt[2], max_pz, power_z);

                            quad_u_values.clear();
                            quad_w_values.clear();
                            quad_u_basis.evaluate_values(Vec3{u, Real(0), Real(0)}, quad_u_values);
                            quad_w_basis.evaluate_values(Vec3{w, Real(0), Real(0)}, quad_w_values);

                            for (std::size_t p = 0; p < modal_count; ++p) {
                                const auto& poly = monomials_[p];
                                Real du = Real(0), dw = Real(0);
                                for (int tt = 0; tt < poly.num_terms; ++tt) {
                                    const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                    const Real mv = power_x[static_cast<std::size_t>(mono.px)] *
                                                    power_y[static_cast<std::size_t>(mono.py)] *
                                                    power_z[static_cast<std::size_t>(mono.pz)];
                                    du += mono.coefficient * tu[static_cast<std::size_t>(mono.component)] * mv;
                                    dw += mono.coefficient * tw[static_cast<std::size_t>(mono.component)] * mv;
                                }
                                mono_u[p] = du;
                                mono_w[p] = dw;
                            }

                            const Real wt = wq * scale;
                            // u-directed moments
                            const std::size_t k_plus_1 = static_cast<std::size_t>(k + 1);
                            for (std::size_t a = 0; a < quad_u_values.size(); ++a) {
                                for (std::size_t b = 0; b < quad_w_values.size(); ++b) {
                                    if (a * k_plus_1 + b >= n_u) continue;
                                    const Real wa = wt * quad_u_values[a] * quad_w_values[b];
                                    for (std::size_t p = 0; p < modal_count; ++p) {
                                        A[(row + a * k_plus_1 + b) * n + p] += wa * mono_u[p];
                                    }
                                }
                            }
                        }
                        row += n_u + n_w;
                    }
                }

                // Interior moments for wedge/pyramid
                if (k >= 2) {
                    if (is_wedge(type)) {
                        // Wedge interior: P_{k-2}(x,y) x P_{k-1}(z) for each component
                        for (int c = 0; c < 3; ++c) {
                            for (int l = 0; l <= k - 1; ++l) {
                                for (int j = 0; j <= k - 2; ++j) {
                                    for (int i = 0; i <= k - 2 - j; ++i) {
                                        if (row >= n) break;
                                        for (std::size_t p = 0; p < modal_count; ++p) {
                                            const auto& poly = monomials_[p];
                                            Real acc = Real(0);
                                            for (int tt = 0; tt < poly.num_terms; ++tt) {
                                                const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                                if (mono.component != c) continue;
                                                acc += mono.coefficient *
                                                       integral_triangle_monomial(mono.px + i, mono.py + j) *
                                                       integral_monomial_1d(mono.pz + l);
                                            }
                                            A[row * n + p] = acc;
                                        }
                                        ++row;
                                    }
                                }
                            }
                        }
                    } else {
                        // Pyramid interior: Q_{k-2}(x,y) x P_{k-2}(z)
                        for (int c = 0; c < 3; ++c) {
                            for (int l = 0; l <= k - 2; ++l) {
                                for (int j = 0; j <= k - 2; ++j) {
                                    for (int i = 0; i <= k - 2; ++i) {
                                        if (row >= n) break;
                                        for (std::size_t p = 0; p < modal_count; ++p) {
                                            const auto& poly = monomials_[p];
                                            Real acc = Real(0);
                                            for (int tt = 0; tt < poly.num_terms; ++tt) {
                                                const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                                if (mono.component != c) continue;
                                                acc += mono.coefficient *
                                                       integral_monomial_1d(mono.px + i) *
                                                       integral_monomial_1d(mono.py + j) *
                                                       integral_pyramid_z(mono.pz + l);
                                            }
                                            A[row * n + p] = acc;
                                        }
                                        ++row;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        FE_CHECK_ARG(row == n, "NedelecBasis: DOF assembly did not fill matrix");

        const std::vector<Real> coeffs = invert_dense_matrix(std::move(A), n);
        modal_sparse_coeffs_ = build_sparse_modal_coefficients(coeffs, n, n);
        nodal_generated_ = true;
    }
}


} // namespace basis
} // namespace FE
} // namespace svmp
