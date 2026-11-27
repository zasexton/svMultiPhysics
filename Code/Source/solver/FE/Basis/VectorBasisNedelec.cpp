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
using detail::vector_construction::invert_dense_matrix;
using quadrature::reference_integrals::integral_monomial_1d;
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

    if (order_ >= 1 && is_pyramid(type)) {
        auto candidates = make_nd_extra_monomial_candidates(type, order_);
        const auto coeffs = build_nd_direct_transform(type, order_, size_, candidates);
        for (const auto& m : candidates) {
            ModalPolynomial p;
            p.num_terms = 1;
            p.terms[0] = {m[0], m[1], m[2], m[3], Real(1)};
            monomials_.push_back(p);
        }
        modal_sparse_coeffs_ = build_sparse_modal_coefficients(coeffs, monomials_.size(), size_);
        modal_power_limits_ = modal_power_limits(monomials_);
        nodal_generated_ = true;
        return;
    }

    if (order_ > 0) {
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
                monomials_ = wedge_vector_modes(k, BasisType::Nedelec);
            }
        }
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

        const auto face_moments = [&](std::size_t face) {
            return detail::vector_construction::add_hcurl_face_moments(type, ref.face_nodes(face),
                k, std::span<Real>(A).subspan(row*n), n, [&](const Vec3& xi, std::vector<Vec3>& values) {
                    fill_powers(xi[0], max_px, power_x);
                    fill_powers(xi[1], max_py, power_y);
                    fill_powers(xi[2], max_pz, power_z);
                    for (std::size_t p = 0; p < n; ++p) {
                        auto& value = values[p];
                        value = {};
                        const auto& polynomial = monomials_[p];
                        for (int t = 0; t < polynomial.num_terms; ++t) {
                            const auto& m = polynomial.terms[t];
                            value[m.component] += m.coefficient * power_x[m.px] * power_y[m.py] * power_z[m.pz];
                        }
                    }
                });
        };

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
                for (std::size_t f = 0; f < ref.num_faces(); ++f) row += face_moments(f);

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
                for (std::size_t f = 0; f < ref.num_faces(); ++f) row += face_moments(f);
                // The component along each axis has degree k on that axis,
                // and two tangential boundary values on each other axis.
                for (int c = 0; c < 3; ++c) {
                    for (int l = 0; l <= (c == 2 ? k : k-1); ++l) {
                        for (int j = 0; j <= (c == 1 ? k : k-1); ++j) {
                            for (int i = 0; i <= (c == 0 ? k : k-1); ++i) {
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
            } else if (is_wedge(type)) {
                const auto modal_count = monomials_.size();
                for (std::size_t f = 0; f < ref.num_faces(); ++f) row += face_moments(f);

                for (int c = 0; c < 3; ++c) {
                    const int tri_degree = c == 2 ? k-2 : k-1;
                    const int line_degree = c == 2 ? k : k-1;
                    for (int l = 0; l <= line_degree; ++l) {
                        for (int j = 0; j <= tri_degree; ++j) {
                            for (int i = 0; i <= tri_degree-j; ++i) {
                                FE_CHECK_ARG(row < n, "Prism interior moment overflow");
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
