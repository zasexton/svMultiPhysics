/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "VectorBasis.h"
#include "Basis/BasisTraits.h"
#include "Math/DenseLinearAlgebra.h"
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
#include <cmath>
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

using detail::vector_construction::build_rt_direct_transform;
using detail::vector_construction::eval_rt_seed_divergence;
using detail::vector_construction::eval_rt_seed_values;
using detail::vector_construction::invert_dense_matrix;
using detail::vector_construction::rank_revealing_pseudo_inverse_dense_matrix;
using quadrature::reference_integrals::integral_monomial_1d;
using quadrature::reference_integrals::integral_pyramid_z;
using quadrature::reference_integrals::integral_tetra_monomial;
using quadrature::reference_integrals::integral_triangle_monomial;
using quadrature::reference_integrals::integral_wedge_monomial;

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

RaviartThomasBasis::RaviartThomasBasis(ElementType type, int order)
    : element_type_(type), dimension_(0), order_(order) {
    if (order_ < 0) {
        throw BasisConfigurationException("RaviartThomasBasis requires non-negative order",
                                          __FILE__, __LINE__, __func__);
    }

    ensure_supported_hybrid_vector_order(type, order_, "RaviartThomasBasis");

    if (is_triangle(type) || is_quadrilateral(type)) {
        dimension_ = 2;
        if (is_triangle(type)) {
            // Simplex RT(k): [P_k]^2 ⊕ x * \tilde P_k (homogeneous degree k)
            const std::size_t k = static_cast<std::size_t>(order_);
            size_ = (k + 1u) * (k + 3u);
        } else {
            // Tensor-product RT(k): Q_{k+1,k} × Q_{k,k+1}
            const std::size_t k = static_cast<std::size_t>(order_);
            size_ = 2u * (k + 1u) * (k + 2u);
        }
    } else if (is_tetrahedron(type)) {
        dimension_ = 3;
        // Simplex RT(k): [P_k]^3 ⊕ x * \tilde P_k (homogeneous degree k)
        const std::size_t k = static_cast<std::size_t>(order_);
        size_ = (k + 1u) * (k + 2u) * (k + 4u) / 2u;
    } else if (is_wedge(type)) {
        dimension_ = 3;
        size_ = rt_wedge_size(order_);
    } else if (is_hexahedron(type)) {
        dimension_ = 3;
        // Tensor-product RT(k): Q_{k+1,k,k} × Q_{k,k+1,k} × Q_{k,k,k+1}
        const std::size_t k = static_cast<std::size_t>(order_);
        size_ = 3u * (k + 1u) * (k + 1u) * (k + 2u);
    } else if (is_pyramid(type)) {
        dimension_ = 3;
        size_ = rt_pyramid_size(order_);
    } else {
        throw BasisElementCompatibilityException("RaviartThomasBasis supports triangles/quadrilaterals (2D) and "
                                                 "tetrahedra/hexahedra/wedges/pyramids (3D)",
                                                 __FILE__, __LINE__, __func__);
    }

    // Wedge/pyramid RT(1-2) uses the explicit seed formulas transformed into a
    // nodal basis with the actual face/interior DOF functionals. For k>=3 we
    // switch to an overcomplete polynomial candidate space and solve the same
    // moment-fitting system against the full hybrid-cell DOF set.
    if (order_ >= 1 && (is_wedge(type) || is_pyramid(type))) {
        transformed_seed_indices_.resize(size_);
        if (order_ <= 2) {
            for (std::size_t i = 0; i < size_; ++i) {
                transformed_seed_indices_[i] = static_cast<int>(i);
            }
            if (is_wedge(type)) {
                transformed_seed_jacobian_evaluator_ =
                    order_ == 1
                        ? &detail::vector_direct::eval_wedge_rt1_jacobians
                        : &detail::vector_direct::eval_wedge_rt2_jacobians;
            } else {
                transformed_seed_jacobian_evaluator_ =
                    order_ == 1
                        ? &detail::vector_direct::eval_pyramid_rt1_jacobians
                        : &detail::vector_direct::eval_pyramid_rt2_jacobians;
            }
        } else {
            transformed_seed_indices_.clear();
        }
        transformed_monomial_candidates_ = make_rt_extra_monomial_candidates(type, order_);
        transformed_power_limits_ = component_monomial_power_limits(transformed_monomial_candidates_);
        const std::vector<Real> coeffs =
            build_rt_direct_transform(type, order_, size_, transformed_monomial_candidates_);
        transformed_sparse_coeffs_ = build_sparse_modal_coefficients(
            coeffs,
            transformed_seed_indices_.size() + transformed_monomial_candidates_.size(),
            size_);
        use_transformed_direct_seed_ = true;
        return;
    }

    // Generate nodal (moment-based) basis functions via DOF matrix inversion.
    // This is used for all supported orders, including k=0, to ensure consistent
    // entity ordering and orientation behavior across mesh permutations.
    if (is_quadrilateral(type) || is_hexahedron(type) || is_triangle(type) ||
        is_tetrahedron(type) || ((is_wedge(type) || is_pyramid(type)) && order_ > 0)) {
        const std::size_t n = size_;

        // ------------------------------------------------------------------
        // Modal monomial basis
        // ------------------------------------------------------------------
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
                // x * \tilde P_k (homogeneous degree k): q_i = x^i y^{k-i}
                for (int i = 0; i <= k; ++i) {
                    ModalPolynomial poly;
                    poly.num_terms = 2;
                    poly.terms[0] = ModalTerm{0, i + 1, k - i, 0, Real(1)};     // x*q
                    poly.terms[1] = ModalTerm{1, i, k - i + 1, 0, Real(1)};     // y*q
                    append_unique_modal_polynomial(monomials_, poly);
                }
            } else {
                // Tensor-product RT(k): Q_{k+1,k} × Q_{k,k+1}
                // x-component: i=0..k+1, j=0..k
                for (int j = 0; j <= k; ++j) {
                    for (int i = 0; i <= k + 1; ++i) {
                        push_single(0, i, j, 0);
                    }
                }
                // y-component: i=0..k, j=0..k+1
                for (int j = 0; j <= k + 1; ++j) {
                    for (int i = 0; i <= k; ++i) {
                        push_single(1, i, j, 0);
                    }
                }
            }
        } else {
            if (is_tetrahedron(type)) {
                // [P_k]^3
                for (int l = 0; l <= k; ++l) {
                    for (int j = 0; j <= k - l; ++j) {
                        for (int i = 0; i <= k - l - j; ++i) {
                            push_single(0, i, j, l);
                        }
                    }
                }
                for (int l = 0; l <= k; ++l) {
                    for (int j = 0; j <= k - l; ++j) {
                        for (int i = 0; i <= k - l - j; ++i) {
                            push_single(1, i, j, l);
                        }
                    }
                }
                for (int l = 0; l <= k; ++l) {
                    for (int j = 0; j <= k - l; ++j) {
                        for (int i = 0; i <= k - l - j; ++i) {
                            push_single(2, i, j, l);
                        }
                    }
                }
                // x * \tilde P_k (homogeneous degree k): q_{i,j} = x^i y^j z^{k-i-j}
                for (int j = 0; j <= k; ++j) {
                    for (int i = 0; i <= k - j; ++i) {
                        const int l = k - i - j;
                        ModalPolynomial poly;
                        poly.num_terms = 3;
                        poly.terms[0] = ModalTerm{0, i + 1, j, l, Real(1)}; // x*q
                        poly.terms[1] = ModalTerm{1, i, j + 1, l, Real(1)}; // y*q
                        poly.terms[2] = ModalTerm{2, i, j, l + 1, Real(1)}; // z*q
                        append_unique_modal_polynomial(monomials_, poly);
                    }
                }
            } else if (is_hexahedron(type)) {
                // Tensor-product RT(k): Q_{k+1,k,k} × Q_{k,k+1,k} × Q_{k,k,k+1}
                // x-component: i=0..k+1, j=0..k, l=0..k
                for (int l = 0; l <= k; ++l) {
                    for (int j = 0; j <= k; ++j) {
                        for (int i = 0; i <= k + 1; ++i) {
                            push_single(0, i, j, l);
                        }
                    }
                }
                // y-component: i=0..k, j=0..k+1, l=0..k
                for (int l = 0; l <= k; ++l) {
                    for (int j = 0; j <= k + 1; ++j) {
                        for (int i = 0; i <= k; ++i) {
                            push_single(1, i, j, l);
                        }
                    }
                }
                // z-component: i=0..k, j=0..k, l=0..k+1
                for (int l = 0; l <= k + 1; ++l) {
                    for (int j = 0; j <= k; ++j) {
                        for (int i = 0; i <= k; ++i) {
                            push_single(2, i, j, l);
                        }
                    }
                }
            } else if (is_wedge(type)) {
                // ================================================================
                // Wedge RT(k): prismatic H(div) element - CORRECTED CONSTRUCTION
                // ================================================================
                //
                // Reference: Bergot & Duruffle, "Approximation of H(div) with high-order
                // optimal finite elements for pyramids, prisms and hexahedra," 2013.
                //
                // The RT_k space on a prism is constructed using a HIERARCHICAL approach
                // that explicitly separates face-based and interior-based functions.
                //
                // KEY INSIGHT: Instead of building a general polynomial space and truncating,
                // we build basis functions that correspond directly to the DOF structure:
                //
                // 1. FACE BASIS FUNCTIONS (face-normal polynomials):
                //    - Triangular faces: lift P_k(triangle) normal-flux functions
                //    - Quad faces: lift Q_k(quad) normal-flux functions
                //
                // 2. INTERIOR BASIS FUNCTIONS (bubble functions):
                //    - Divergence-compatible interior polynomials
                //
                // This construction guarantees that the moment matrix is block-structured
                // and well-conditioned.
                //
                // For RT(0): only face functions exist (5 DOFs)
                // For RT(k), k>=1: face + interior functions
                //
                // We use a two-phase approach:
                // Phase 1: Build face-associated modal functions
                // Phase 2: Build interior modal functions

                // --- Phase 1: Face-associated basis functions ---

                // Bottom triangular face (z = -1): normal = (0, 0, -1)
                // Functions: P_k(x,y) * z-dependent factor that localizes to bottom
                // We use: v = (0, 0, -q(x,y) * (1-z)/2) for q in P_k(tri)
                for (int j = 0; j <= k; ++j) {
                    for (int i = 0; i <= k - j; ++i) {
                        // -q(x,y) * (1-z)/2 = -0.5*q + 0.5*q*z
                        ModalPolynomial poly;
                        poly.num_terms = 2;
                        poly.terms[0] = ModalTerm{2, i, j, 0, Real(-0.5)};
                        poly.terms[1] = ModalTerm{2, i, j, 1, Real(0.5)};
                        append_unique_modal_polynomial(monomials_, poly);
                    }
                }

                // Top triangular face (z = +1): normal = (0, 0, +1)
                // Functions: v = (0, 0, q(x,y) * (1+z)/2) for q in P_k(tri)
                for (int j = 0; j <= k; ++j) {
                    for (int i = 0; i <= k - j; ++i) {
                        // q(x,y) * (1+z)/2 = 0.5*q + 0.5*q*z
                        ModalPolynomial poly;
                        poly.num_terms = 2;
                        poly.terms[0] = ModalTerm{2, i, j, 0, Real(0.5)};
                        poly.terms[1] = ModalTerm{2, i, j, 1, Real(0.5)};
                        append_unique_modal_polynomial(monomials_, poly);
                    }
                }

                // Quad face 0 (y = 0): normal = (0, -1, 0)
                // Functions: v = (0, -q_xz(x,z) * (1-y), 0) for q in Q_k
                for (int l = 0; l <= k; ++l) {
                    for (int i = 0; i <= k; ++i) {
                        // -q(x,z) * (1-y) = -q + q*y
                        ModalPolynomial poly;
                        poly.num_terms = 2;
                        poly.terms[0] = ModalTerm{1, i, 0, l, Real(-1)};
                        poly.terms[1] = ModalTerm{1, i, 1, l, Real(1)};
                        append_unique_modal_polynomial(monomials_, poly);
                    }
                }

                // Quad face 1 (x = 0): normal = (-1, 0, 0)
                // Functions: v = (-q_yz(y,z) * (1-x), 0, 0) for q in Q_k
                for (int l = 0; l <= k; ++l) {
                    for (int j = 0; j <= k; ++j) {
                        // -q(y,z) * (1-x) = -q + q*x
                        ModalPolynomial poly;
                        poly.num_terms = 2;
                        poly.terms[0] = ModalTerm{0, 0, j, l, Real(-1)};
                        poly.terms[1] = ModalTerm{0, 1, j, l, Real(1)};
                        append_unique_modal_polynomial(monomials_, poly);
                    }
                }

                // Quad face 2 (x + y = 1): normal = (1/sqrt(2), 1/sqrt(2), 0)
                // This face is more complex. We use functions that have the correct
                // normal flux on this face.
                // Functions: v = (q(s,z), q(s,z), 0) * localization factor
                // where s is the face parameter and localization = (x+y)
                for (int l = 0; l <= k; ++l) {
                    for (int s = 0; s <= k; ++s) {
                        // For face x+y=1, parameterize by t=x, z
                        // Normal flux is proportional to v_x + v_y
                        // Use v = ((x+y)^s * z^l, (x+y)^s * z^l, 0) * (x+y - (1-x-y))
                        //       = ((x+y)^s * z^l, (x+y)^s * z^l, 0) * (2(x+y) - 1)
                        // Simplified: use x^s*z^l + y^s*z^l for both components
                        ModalPolynomial poly;
                        poly.num_terms = 4;
                        poly.terms[0] = ModalTerm{0, s, 0, l, Real(1)};
                        poly.terms[1] = ModalTerm{0, 0, s, l, Real(1)};
                        poly.terms[2] = ModalTerm{1, s, 0, l, Real(1)};
                        poly.terms[3] = ModalTerm{1, 0, s, l, Real(1)};
                        append_unique_modal_polynomial(monomials_, poly);
                    }
                }

                // --- Phase 2: Interior basis functions (for k >= 1) ---
                // Interior functions have zero normal flux on all faces
                // These are the "bubble" functions in H(div)

                if (k >= 1) {
                    // Interior basis for RT(k) on prism:
                    // The interior DOFs test against P_{k-1}(tri) x P_k(z) for each component.
                    // dim(P_{k-1}(tri)) = k(k+1)/2
                    // dim(P_k(z)) = k+1
                    // Total interior DOFs = 3 * k(k+1)/2 * (k+1) = 3k(k+1)^2/2
                    //
                    // For k=1: 3 * 1 * 1 * 2 = 6 interior DOFs
                    // Each component has k(k+1)/2 * (k+1) = k(k+1)^2/2 functions
                    // For k=1: 1 * 2 = 2 functions per component

                    // Interior x-component: P_{k-1}(x,y) x P_k(z)
                    for (int l = 0; l <= k; ++l) {
                        for (int j = 0; j <= k - 1; ++j) {
                            for (int i = 0; i <= k - 1 - j; ++i) {
                                push_single(0, i, j, l);
                            }
                        }
                    }
                    // Interior y-component: P_{k-1}(x,y) x P_k(z)
                    for (int l = 0; l <= k; ++l) {
                        for (int j = 0; j <= k - 1; ++j) {
                            for (int i = 0; i <= k - 1 - j; ++i) {
                                push_single(1, i, j, l);
                            }
                        }
                    }
                    // Interior z-component: P_{k-1}(x,y) x P_k(z)
                    for (int l = 0; l <= k; ++l) {
                        for (int j = 0; j <= k - 1; ++j) {
                            for (int i = 0; i <= k - 1 - j; ++i) {
                                push_single(2, i, j, l);
                            }
                        }
                    }
                }

                // If we have more modal functions than DOFs, we need to select
                // a linearly independent subset. This will be handled by the
                // DOF matrix construction with column pivoting.
                // For now, mark that we may have an oversized basis.
                // The DOF assembly code will handle this.
            } else if (is_pyramid(type)) {
                // ================================================================
                // Pyramid RT(k): H(div) element on pyramid - CORRECTED CONSTRUCTION
                // ================================================================
                //
                // Reference: Nigam-Phillips (2012), Falk-Gatto-Monk (2011)
                //
                // The pyramid has 1 quad face (base z=0) and 4 triangular faces.
                // The construction uses a hierarchical face-based + interior approach.
                //
                // For k=0: 5 face DOFs only
                // For k>=1: face DOFs + interior DOFs
                //
                // Face DOFs:
                //   - Quad face (base): (k+1)^2 DOFs
                //   - 4 triangular faces: 4 * (k+1)(k+2)/2 DOFs
                //
                // Interior DOFs: 3 * k^3 for k>=1 (pyramid-specific)
                //
                // We use hierarchical construction similar to wedge.

                // --- Phase 1: Face-associated basis functions ---

                // Quad base face (z = 0): normal = (0, 0, -1)
                // Functions: v = (0, 0, -q(x,y) * (1-z)) for q in Q_k
                for (int j = 0; j <= k; ++j) {
                    for (int i = 0; i <= k; ++i) {
                        // -q(x,y) * (1-z) = -q + q*z
                        ModalPolynomial poly;
                        poly.num_terms = 2;
                        poly.terms[0] = ModalTerm{2, i, j, 0, Real(-1)};
                        poly.terms[1] = ModalTerm{2, i, j, 1, Real(1)};
                        append_unique_modal_polynomial(monomials_, poly);
                    }
                }

                // Triangular face 0 (x = -1+z, y free): normal roughly (-1,0,1)/sqrt(2)
                // We parameterize using face coordinates and lift
                for (int j = 0; j <= k; ++j) {
                    for (int i = 0; i <= k - j; ++i) {
                        // Use a localized function based on face geometry
                        // Simple approach: use (x+1-z)-weighted functions
                        ModalPolynomial poly;
                        poly.num_terms = 2;
                        poly.terms[0] = ModalTerm{0, i, j, 0, Real(-1)};
                        poly.terms[1] = ModalTerm{0, i, j, 1, Real(1)};
                        append_unique_modal_polynomial(monomials_, poly);
                    }
                }

                // Triangular face 1 (x = 1-z, y free): normal roughly (1,0,1)/sqrt(2)
                for (int j = 0; j <= k; ++j) {
                    for (int i = 0; i <= k - j; ++i) {
                        ModalPolynomial poly;
                        poly.num_terms = 2;
                        poly.terms[0] = ModalTerm{0, i, j, 0, Real(1)};
                        poly.terms[1] = ModalTerm{0, i, j, 1, Real(1)};
                        append_unique_modal_polynomial(monomials_, poly);
                    }
                }

                // Triangular face 2 (y = -1+z, x free): normal roughly (0,-1,1)/sqrt(2)
                for (int i = 0; i <= k; ++i) {
                    for (int j = 0; j <= k - i; ++j) {
                        ModalPolynomial poly;
                        poly.num_terms = 2;
                        poly.terms[0] = ModalTerm{1, i, j, 0, Real(-1)};
                        poly.terms[1] = ModalTerm{1, i, j, 1, Real(1)};
                        append_unique_modal_polynomial(monomials_, poly);
                    }
                }

                // Triangular face 3 (y = 1-z, x free): normal roughly (0,1,1)/sqrt(2)
                for (int i = 0; i <= k; ++i) {
                    for (int j = 0; j <= k - i; ++j) {
                        ModalPolynomial poly;
                        poly.num_terms = 2;
                        poly.terms[0] = ModalTerm{1, i, j, 0, Real(1)};
                        poly.terms[1] = ModalTerm{1, i, j, 1, Real(1)};
                        append_unique_modal_polynomial(monomials_, poly);
                    }
                }

                // --- Phase 2: Interior basis functions (for k >= 1) ---
                if (k >= 1) {
                    // Interior DOFs for pyramid: 3 * k^3
                    // Each component has k^3 interior functions
                    // Test space: P_{k-1}(x,y) x P_{k-1}(z) for each component

                    // Interior x-component: P_{k-1}(x,y) x P_{k-1}(z)
                    for (int l = 0; l <= k - 1; ++l) {
                        for (int j = 0; j <= k - 1; ++j) {
                            for (int i = 0; i <= k - 1; ++i) {
                                push_single(0, i, j, l);
                            }
                        }
                    }
                    // Interior y-component: P_{k-1}(x,y) x P_{k-1}(z)
                    for (int l = 0; l <= k - 1; ++l) {
                        for (int j = 0; j <= k - 1; ++j) {
                            for (int i = 0; i <= k - 1; ++i) {
                                push_single(1, i, j, l);
                            }
                        }
                    }
                    // Interior z-component: P_{k-1}(x,y) x P_{k-1}(z)
                    for (int l = 0; l <= k - 1; ++l) {
                        for (int j = 0; j <= k - 1; ++j) {
                            for (int i = 0; i <= k - 1; ++i) {
                                push_single(2, i, j, l);
                            }
                        }
                    }
                }
            }
        }
        // For wedge and pyramid elements, we may have an oversized modal basis.
        // We will use column selection during DOF matrix assembly.
        const std::size_t m = monomials_.size(); // number of modal functions (may be > n)
        const bool oversized_basis = (m > n);

        // For non-oversized cases, verify exact match
        if (!oversized_basis) {
            FE_CHECK_ARG(m == n,
                         "RaviartThomasBasis: modal basis size mismatch (expected " +
                         std::to_string(n) + ", got " + std::to_string(monomials_.size()) + ")");
        }

        // ------------------------------------------------------------------
        // Assemble DOF matrix for classical RT(k) moments and invert
        // ------------------------------------------------------------------
        auto assembly_power_limits = modal_power_limits(monomials_);
        const int max_px = assembly_power_limits[0];
        const int max_py = assembly_power_limits[1];
        const int max_pz = assembly_power_limits[2];
        std::vector<Real> power_x;
        std::vector<Real> power_y;
        std::vector<Real> power_z;

        // For oversized basis, we need to select n linearly independent columns
        // For standard cases, m == n and we just invert directly.
        // We'll handle oversized bases at the end by column selection.
        std::vector<Real> A(n * n, Real(0));
        const elements::ReferenceElement ref = elements::ReferenceElement::create(type);
        std::size_t row = 0;

        // If oversized, select the first deterministic, duplicate-pruned
        // candidates. The rank check below verifies that the subset spans the
        // requested DOFs.
        if (oversized_basis) {
            monomials_.erase(monomials_.begin() + static_cast<std::ptrdiff_t>(n),
                             monomials_.end());
            modal_power_limits_ = modal_power_limits(monomials_);
        } else {
            modal_power_limits_ = assembly_power_limits;
        }

        if (dimension_ == 2) {
            // Edge flux moments: ∫_e (v·n) * l_i(s) ds, i=0..k.
            const LagrangeBasis edge_basis(ElementType::Line2, k);
            const auto edge_quad = quadrature::QuadratureFactory::create(
                ElementType::Line2, 2 * k + 2, QuadratureType::GaussLegendre);
            std::vector<Real> edge_values;

            for (std::size_t e = 0; e < ref.num_edges(); ++e) {
                const auto& en = ref.edge_nodes(e);
                FE_CHECK_ARG(en.size() == 2u, "RT quad: expected 2 vertices per edge");
                const Vec3 p0 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(en[0]));
                const Vec3 p1 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(en[1]));
                const Vec3 t = normalize3(p1 - p0);
                const Vec3 nrm{t[1], -t[0], Real(0)}; // rotate -90
                const Real J = math::norm(p1 - p0) * Real(0.5);

                for (int a = 0; a <= k; ++a) {
                    FE_CHECK_ARG(row < n, "RT quad: row overflow in edge moments");
                    for (std::size_t q = 0; q < edge_quad->num_points(); ++q) {
                        const Real s = edge_quad->point(q)[0];
                        const Real wq = edge_quad->weight(q);
                        edge_values.clear();
                        edge_basis.evaluate_values(Vec3{s, Real(0), Real(0)}, edge_values);
                        FE_CHECK_ARG(edge_values.size() == static_cast<std::size_t>(k + 1), "RT quad: edge basis size mismatch");
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
                                dot += mono.coefficient * nrm[static_cast<std::size_t>(mono.component)] * mv;
                            }
                            A[row * n + p] += wt * dot;
                        }
                    }
                    ++row;
                }
            }

            if (is_triangle(type)) {
                // Interior moments: component-wise moments against P_{k-1} monomials on the simplex.
                for (int c = 0; c < 2; ++c) {
                    for (int j = 0; j <= k - 1; ++j) {
                        for (int i = 0; i <= k - 1 - j; ++i) {
                            FE_CHECK_ARG(row < n, "RT triangle: row overflow in interior moments");
                            for (std::size_t p = 0; p < n; ++p) {
                                const auto& poly = monomials_[p];
                                Real acc = Real(0);
                                for (int t = 0; t < poly.num_terms; ++t) {
                                    const auto& mono = poly.terms[static_cast<std::size_t>(t)];
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
                // Interior moments against Q_{k-1,k}×Q_{k,k-1} monomials (tensor-product reference).
                // x-component test space: i=0..k-1, j=0..k
                for (int j = 0; j <= k; ++j) {
                    for (int i = 0; i <= k - 1; ++i) {
                        FE_CHECK_ARG(row < n, "RT quad: row overflow in interior x-moments");
                        for (std::size_t p = 0; p < n; ++p) {
                            const auto& poly = monomials_[p];
                            Real acc = Real(0);
                            for (int t = 0; t < poly.num_terms; ++t) {
                                const auto& mono = poly.terms[static_cast<std::size_t>(t)];
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
                // y-component test space: i=0..k, j=0..k-1
                for (int j = 0; j <= k - 1; ++j) {
                    for (int i = 0; i <= k; ++i) {
                        FE_CHECK_ARG(row < n, "RT quad: row overflow in interior y-moments");
                        for (std::size_t p = 0; p < n; ++p) {
                            const auto& poly = monomials_[p];
                            Real acc = Real(0);
                            for (int t = 0; t < poly.num_terms; ++t) {
                                const auto& mono = poly.terms[static_cast<std::size_t>(t)];
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
                // Face flux moments: ∫_f (v·n) * l_a(u,v) dS, a in P_k(face).
                const LagrangeBasis face_basis(ElementType::Triangle3, k);
                const auto face_quad = quadrature::QuadratureFactory::create(
                    ElementType::Triangle3, 2 * k + 2, QuadratureType::GaussLegendre);

                const std::size_t nface = face_basis.size();
                std::vector<Real> face_values;
                std::vector<Real> modal_dot(n, Real(0));
                for (std::size_t f = 0; f < ref.num_faces(); ++f) {
                    const auto& fn = ref.face_nodes(f);
                    FE_CHECK_ARG(fn.size() == 3u, "RT tetra: expected tri face with 3 vertices");
                    const Vec3 v0 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[0]));
                    const Vec3 v1 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[1]));
                    const Vec3 v2 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[2]));
                    const Vec3 e01 = v1 - v0;
                    const Vec3 e02 = v2 - v0;
                    const Vec3 nrm = normalize3(cross3(e01, e02));
                    const Real scale = cross3(e01, e02).norm();

                    FE_CHECK_ARG(row + nface <= n, "RT tetra: row overflow in face moments");

                    for (std::size_t q = 0; q < face_quad->num_points(); ++q) {
                        const auto uv = face_quad->point(q);
                        const Real u = uv[0];
                        const Real v = uv[1];
                        const Real wq = face_quad->weight(q);

                        face_values.clear();
                        face_basis.evaluate_values(Vec3{u, v, Real(0)}, face_values);
                        FE_CHECK_ARG(face_values.size() == nface, "RT tetra: face basis size mismatch");

                        const Vec3 xi = v0 + e01 * u + e02 * v;
                        fill_powers(xi[0], max_px, power_x);
                        fill_powers(xi[1], max_py, power_y);
                        fill_powers(xi[2], max_pz, power_z);

                        for (std::size_t p = 0; p < n; ++p) {
                            const auto& poly = monomials_[p];
                            Real dot = Real(0);
                            for (int t = 0; t < poly.num_terms; ++t) {
                                const auto& mono = poly.terms[static_cast<std::size_t>(t)];
                                const Real mv =
                                    power_x[static_cast<std::size_t>(mono.px)] *
                                    power_y[static_cast<std::size_t>(mono.py)] *
                                    power_z[static_cast<std::size_t>(mono.pz)];
                                dot += mono.coefficient * nrm[static_cast<std::size_t>(mono.component)] * mv;
                            }
                            modal_dot[p] = dot;
                        }

                        const Real wt = wq * scale;
                        for (std::size_t a = 0; a < nface; ++a) {
                            const Real wa = wt * face_values[a];
                            if (wa == Real(0)) {
                                continue;
                            }
                            const std::size_t r = row + a;
                            for (std::size_t p = 0; p < n; ++p) {
                                A[r * n + p] += wa * modal_dot[p];
                            }
                        }
                    }
                    row += nface;
                }

                // Interior moments: component-wise moments against P_{k-1} monomials on the simplex.
                for (int c = 0; c < 3; ++c) {
                    for (int l = 0; l <= k - 1; ++l) {
                        for (int j = 0; j <= k - 1 - l; ++j) {
                            for (int i = 0; i <= k - 1 - l - j; ++i) {
                                FE_CHECK_ARG(row < n, "RT tetra: row overflow in interior moments");
                                for (std::size_t p = 0; p < n; ++p) {
                                    const auto& poly = monomials_[p];
                                    Real acc = Real(0);
                                    for (int t = 0; t < poly.num_terms; ++t) {
                                        const auto& mono = poly.terms[static_cast<std::size_t>(t)];
                                        if (mono.component != c) {
                                            continue;
                                        }
                                        acc += mono.coefficient *
                                               integral_tetra_monomial(mono.px + i, mono.py + j, mono.pz + l);
                                    }
                                    A[row * n + p] = acc;
                                }
                                ++row;
                            }
                        }
                    }
                }
            } else if (is_hexahedron(type)) {
                // Face flux moments: ∫_f (v·n) * l_a(u,v) dS, a in Q_k.
                const LagrangeBasis face_basis(ElementType::Quad4, k);
                const auto face_quad = quadrature::QuadratureFactory::create(
                    ElementType::Quad4, 2 * k + 2, QuadratureType::GaussLegendre);
                std::vector<Real> face_values;
                std::vector<Real> modal_dot(n, Real(0));

                for (std::size_t f = 0; f < ref.num_faces(); ++f) {
                    const auto& fn = ref.face_nodes(f);
                    FE_CHECK_ARG(fn.size() == 4u, "RT hex: expected quad face with 4 vertices");
                    const std::array<Vec3, 4> fv{
                        ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[0])),
                        ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[1])),
                        ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[2])),
                        ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[3]))
                    };
                    const Vec3 nrm = normalize3(cross3(fv[1] - fv[0], fv[3] - fv[0]));

                    const std::size_t nface = static_cast<std::size_t>((k + 1) * (k + 1));
                    FE_CHECK_ARG(row + nface <= n, "RT hex: row overflow in face moments");

                    for (std::size_t q = 0; q < face_quad->num_points(); ++q) {
                        const auto uv = face_quad->point(q);
                        const Real u = uv[0];
                        const Real w = uv[1];
                        const Real wq = face_quad->weight(q);

                        face_values.clear();
                        face_basis.evaluate_values(Vec3{u, w, Real(0)}, face_values);
                        FE_CHECK_ARG(face_values.size() == nface, "RT hex: face basis size mismatch");

                        const Vec3 xi = bilinear(fv, u, w);
                        const Vec3 dxdu = bilinear_du(fv, u, w);
                        const Vec3 dxdw = bilinear_dw(fv, u, w);
                        const Real scale = cross3(dxdu, dxdw).norm();

                        fill_powers(xi[0], max_px, power_x);
                        fill_powers(xi[1], max_py, power_y);
                        fill_powers(xi[2], max_pz, power_z);

                        for (std::size_t p = 0; p < n; ++p) {
                            const auto& poly = monomials_[p];
                            Real dot = Real(0);
                            for (int t = 0; t < poly.num_terms; ++t) {
                                const auto& mono = poly.terms[static_cast<std::size_t>(t)];
                                const Real mv = power_x[static_cast<std::size_t>(mono.px)] *
                                                power_y[static_cast<std::size_t>(mono.py)] *
                                                power_z[static_cast<std::size_t>(mono.pz)];
                                dot += mono.coefficient * nrm[static_cast<std::size_t>(mono.component)] * mv;
                            }
                            modal_dot[p] = dot;
                        }

                        const Real wt = wq * scale;
                        for (std::size_t a = 0; a < nface; ++a) {
                            const Real wa = wt * face_values[a];
                            if (wa == Real(0)) {
                                continue;
                            }
                            for (std::size_t p = 0; p < n; ++p) {
                                A[(row + a) * n + p] += wa * modal_dot[p];
                            }
                        }
                    }

                    row += nface;
                }

                // Interior moments against Q_{k-1,k,k}×Q_{k,k-1,k}×Q_{k,k,k-1} monomials.
                // x-component: i=0..k-1, j=0..k, l=0..k
                for (int l = 0; l <= k; ++l) {
                    for (int j = 0; j <= k; ++j) {
                        for (int i = 0; i <= k - 1; ++i) {
                            FE_CHECK_ARG(row < n, "RT hex: row overflow in interior x-moments");
                            for (std::size_t p = 0; p < n; ++p) {
                                const auto& poly = monomials_[p];
                                Real acc = Real(0);
                                for (int t = 0; t < poly.num_terms; ++t) {
                                    const auto& mono = poly.terms[static_cast<std::size_t>(t)];
                                    if (mono.component != 0) {
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
                // y-component: i=0..k, j=0..k-1, l=0..k
                for (int l = 0; l <= k; ++l) {
                    for (int j = 0; j <= k - 1; ++j) {
                        for (int i = 0; i <= k; ++i) {
                            FE_CHECK_ARG(row < n, "RT hex: row overflow in interior y-moments");
                            for (std::size_t p = 0; p < n; ++p) {
                                const auto& poly = monomials_[p];
                                Real acc = Real(0);
                                for (int t = 0; t < poly.num_terms; ++t) {
                                    const auto& mono = poly.terms[static_cast<std::size_t>(t)];
                                    if (mono.component != 1) {
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
                // z-component: i=0..k, j=0..k, l=0..k-1
                for (int l = 0; l <= k - 1; ++l) {
                    for (int j = 0; j <= k; ++j) {
                        for (int i = 0; i <= k; ++i) {
                            FE_CHECK_ARG(row < n, "RT hex: row overflow in interior z-moments");
                            for (std::size_t p = 0; p < n; ++p) {
                                const auto& poly = monomials_[p];
                                Real acc = Real(0);
                                for (int t = 0; t < poly.num_terms; ++t) {
                                    const auto& mono = poly.terms[static_cast<std::size_t>(t)];
                                    if (mono.component != 2) {
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
            } else if (is_wedge(type) || is_pyramid(type)) {
                // Mixed face types: wedge has 2 tri + 3 quad faces, pyramid has 1 quad + 4 tri faces
                // We need to handle each face type appropriately.
                const std::size_t modal_count = monomials_.size();

                const LagrangeBasis tri_face_basis(ElementType::Triangle3, k);
                const LagrangeBasis quad_face_basis(ElementType::Quad4, k);
                const auto tri_quad = quadrature::QuadratureFactory::create(
                    ElementType::Triangle3, 2 * k + 2, QuadratureType::GaussLegendre);
                const auto quad_quad = quadrature::QuadratureFactory::create(
                    ElementType::Quad4, 2 * k + 2, QuadratureType::GaussLegendre);
                std::vector<Real> face_values;
                std::vector<Real> modal_dot(modal_count, Real(0));

                for (std::size_t f = 0; f < ref.num_faces(); ++f) {
                    const auto& fn = ref.face_nodes(f);
                    const bool is_tri = (fn.size() == 3u);

                    if (is_tri) {
                        // Triangular face
                        const Vec3 v0 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[0]));
                        const Vec3 v1 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[1]));
                        const Vec3 v2 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[2]));
                        const Vec3 e01 = v1 - v0;
                        const Vec3 e02 = v2 - v0;
                        const Vec3 nrm = normalize3(cross3(e01, e02));
                        const Real scale = cross3(e01, e02).norm();

                        const std::size_t nface = tri_face_basis.size();
                        if (row + nface > n) break; // Safety check

                        for (std::size_t q = 0; q < tri_quad->num_points(); ++q) {
                            const auto uv = tri_quad->point(q);
                            const Real u = uv[0];
                            const Real v = uv[1];
                            const Real wq = tri_quad->weight(q);

                            face_values.clear();
                            tri_face_basis.evaluate_values(Vec3{u, v, Real(0)}, face_values);

                            const Vec3 xi_pt = v0 + e01 * u + e02 * v;
                            fill_powers(xi_pt[0], max_px, power_x);
                            fill_powers(xi_pt[1], max_py, power_y);
                            fill_powers(xi_pt[2], max_pz, power_z);

                            for (std::size_t p = 0; p < modal_count; ++p) {
                                const auto& poly = monomials_[p];
                                Real dot = Real(0);
                                for (int t = 0; t < poly.num_terms; ++t) {
                                    const auto& mono = poly.terms[static_cast<std::size_t>(t)];
                                    const Real mv =
                                        power_x[static_cast<std::size_t>(mono.px)] *
                                        power_y[static_cast<std::size_t>(mono.py)] *
                                        power_z[static_cast<std::size_t>(mono.pz)];
                                    dot += mono.coefficient * nrm[static_cast<std::size_t>(mono.component)] * mv;
                                }
                                modal_dot[p] = dot;
                            }

                            const Real wt = wq * scale;
                            for (std::size_t a = 0; a < nface; ++a) {
                                const Real wa = wt * face_values[a];
                                if (wa == Real(0)) continue;
                                const std::size_t r = row + a;
                                for (std::size_t p = 0; p < modal_count; ++p) {
                                    A[r * n + p] += wa * modal_dot[p];
                                }
                            }
                        }
                        row += nface;
                    } else {
                        // Quad face
                        const std::array<Vec3, 4> fv{
                            ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[0])),
                            ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[1])),
                            ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[2])),
                            ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[3]))
                        };
                        const Vec3 nrm = normalize3(cross3(fv[1] - fv[0], fv[3] - fv[0]));

                        const std::size_t nface = static_cast<std::size_t>((k + 1) * (k + 1));
                        if (row + nface > n) break; // Safety check

                        for (std::size_t q = 0; q < quad_quad->num_points(); ++q) {
                            const auto uv = quad_quad->point(q);
                            const Real u = uv[0];
                            const Real w = uv[1];
                            const Real wq = quad_quad->weight(q);

                            face_values.clear();
                            quad_face_basis.evaluate_values(Vec3{u, w, Real(0)}, face_values);

                            const Vec3 xi_pt = bilinear(fv, u, w);
                            const Vec3 dxdu = bilinear_du(fv, u, w);
                            const Vec3 dxdw = bilinear_dw(fv, u, w);
                            const Real scale = cross3(dxdu, dxdw).norm();

                            fill_powers(xi_pt[0], max_px, power_x);
                            fill_powers(xi_pt[1], max_py, power_y);
                            fill_powers(xi_pt[2], max_pz, power_z);

                            for (std::size_t p = 0; p < modal_count; ++p) {
                                const auto& poly = monomials_[p];
                                Real dot = Real(0);
                                for (int t = 0; t < poly.num_terms; ++t) {
                                    const auto& mono = poly.terms[static_cast<std::size_t>(t)];
                                    const Real mv = power_x[static_cast<std::size_t>(mono.px)] *
                                                    power_y[static_cast<std::size_t>(mono.py)] *
                                                    power_z[static_cast<std::size_t>(mono.pz)];
                                    dot += mono.coefficient * nrm[static_cast<std::size_t>(mono.component)] * mv;
                                }
                                modal_dot[p] = dot;
                            }

                            const Real wt = wq * scale;
                            for (std::size_t a = 0; a < nface; ++a) {
                                const Real wa = wt * face_values[a];
                                if (wa == Real(0)) continue;
                                for (std::size_t p = 0; p < modal_count; ++p) {
                                    A[(row + a) * n + p] += wa * modal_dot[p];
                                }
                            }
                        }
                        row += nface;
                    }
                }

                // Interior moments for wedge/pyramid
                // Use tensor-product approach: prismatic interior for wedge, pyramid interior for pyramid
                if (k >= 1) {
                    if (is_wedge(type)) {
                        // Wedge interior: P_{k-1}(x,y) x P_k(z) for each component
                        for (int c = 0; c < 3; ++c) {
                            for (int l = 0; l <= k; ++l) {
                                for (int j = 0; j <= k - 1; ++j) {
                                    for (int i = 0; i <= k - 1 - j; ++i) {
                                        if (row >= n) break;
                                        for (std::size_t p = 0; p < modal_count; ++p) {
                                            const auto& poly = monomials_[p];
                                            Real acc = Real(0);
                                            for (int t = 0; t < poly.num_terms; ++t) {
                                                const auto& mono = poly.terms[static_cast<std::size_t>(t)];
                                                if (mono.component != c) continue;
                                                // Integral over wedge reference domain
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
                        // Pyramid interior: Q_{k-1}(x,y) x P_{k-1}(z) for each component
                        for (int c = 0; c < 3; ++c) {
                            for (int l = 0; l <= k - 1; ++l) {
                                for (int j = 0; j <= k - 1; ++j) {
                                    for (int i = 0; i <= k - 1; ++i) {
                                        if (row >= n) break;
                                        for (std::size_t p = 0; p < modal_count; ++p) {
                                            const auto& poly = monomials_[p];
                                            Real acc = Real(0);
                                            for (int t = 0; t < poly.num_terms; ++t) {
                                                const auto& mono = poly.terms[static_cast<std::size_t>(t)];
                                                if (mono.component != c) continue;
                                                // Integral over pyramid reference domain approximation
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

        FE_CHECK_ARG(row == n, "RaviartThomasBasis: DOF assembly did not fill matrix");
        if (oversized_basis) {
            const std::size_t rank =
                math::dense_matrix_rank(std::vector<Real>(A.begin(), A.end()), n, n);
            FE_CHECK_ARG(rank == n,
                         "RaviartThomasBasis: oversized modal truncation is rank-deficient "
                         "(rank " + std::to_string(rank) + " of " + std::to_string(n) + ")");
        }

        const std::vector<Real> coeffs =
            (order_ == 1 && (is_wedge(type) || is_pyramid(type)))
                ? rank_revealing_pseudo_inverse_dense_matrix(A, n)
                : invert_dense_matrix(std::move(A), n);
        modal_sparse_coeffs_ = build_sparse_modal_coefficients(coeffs, n, n);
        nodal_generated_ = true;
    }
}

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
