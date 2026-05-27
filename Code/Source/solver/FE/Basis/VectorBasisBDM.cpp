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
#include "VectorBasisRtConstruction.h"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#ifdef FE_CHECK_ARG
#undef FE_CHECK_ARG
#endif
#define FE_CHECK_ARG(condition, message) BASIS_CHECK_CONSTRUCTION((condition), (message))

#include "VectorBasisEvaluationHelpers.h"

namespace svmp {
namespace FE {
namespace basis {

using namespace detail::vector_common;

using detail::vector_construction::invert_dense_matrix;

namespace {

std::shared_ptr<const NedelecBasis> cached_bdm_inner_nedelec_basis(
    ElementType type,
    int order) {
    static std::mutex cache_mutex;
    static std::map<std::pair<ElementType, int>, std::shared_ptr<const NedelecBasis>> cache;

    const auto key = std::make_pair(type, order);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }

    auto basis = std::make_shared<NedelecBasis>(type, order);
    const auto inserted = cache.emplace(key, std::move(basis));
    return inserted.first->second;
}

} // namespace

// ----------------------------------------------------------------------------- //

BDMBasis::BDMBasis(ElementType type, int order)
    : element_type_(type), dimension_(0), order_(order), size_(0) {
    if (order_ < 0) {
        throw BasisConfigurationException("BDMBasis requires non-negative order",
                                          __FILE__, __LINE__, __func__);
    }
    if (order_ == 0) {
        throw BasisConfigurationException("BDMBasis requires order >= 1",
                                          __FILE__, __LINE__, __func__);
    }

    if (type == ElementType::Quad4 || type == ElementType::Quad8 || type == ElementType::Quad9) {
        if (order_ != 1) {
            throw NotImplementedException("BDMBasis quadrilateral support currently requires order 1",
                                          __FILE__, __LINE__, __func__);
        }
        dimension_ = 2;
        size_ = std::size_t(8);
        return;
    }

    if (type == ElementType::Triangle3) {
        dimension_ = 2;
        if (order_ == 1) {
            // Preserve the original BDM1 triangle basis so existing edge-normal
            // traces and divergence normalization remain unchanged.
            size_ = std::size_t(6);
            return;
        }
        const std::size_t k = static_cast<std::size_t>(order_);
        size_ = 2u * triangle_poly_dim(k);
    } else if (type == ElementType::Tetra4) {
        dimension_ = 3;
        const std::size_t k = static_cast<std::size_t>(order_);
        size_ = 3u * tetra_poly_dim(k);
    } else {
        throw BasisElementCompatibilityException("BDMBasis currently supports Triangle3, Tetra4, and quadrilateral variants (Quad4/8/9)",
                                                 __FILE__, __LINE__, __func__);
    }

    const std::size_t n = size_;
    monomials_.clear();
    monomials_.reserve(n);

    auto push_single = [&](int component, int px, int py, int pz) {
        ModalPolynomial poly;
        poly.num_terms = 1;
        poly.terms[0] = ModalTerm{component, px, py, pz, Real(1)};
        monomials_.push_back(poly);
    };

    if (type == ElementType::Triangle3) {
        for (int j = 0; j <= order_; ++j) {
            for (int i = 0; i <= order_ - j; ++i) {
                push_single(0, i, j, 0);
            }
        }
        for (int j = 0; j <= order_; ++j) {
            for (int i = 0; i <= order_ - j; ++i) {
                push_single(1, i, j, 0);
            }
        }
    } else {
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_ - k; ++j) {
                for (int i = 0; i <= order_ - j - k; ++i) {
                    push_single(0, i, j, k);
                }
            }
        }
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_ - k; ++j) {
                for (int i = 0; i <= order_ - j - k; ++i) {
                    push_single(1, i, j, k);
                }
            }
        }
        for (int k = 0; k <= order_; ++k) {
            for (int j = 0; j <= order_ - k; ++j) {
                for (int i = 0; i <= order_ - j - k; ++i) {
                    push_single(2, i, j, k);
                }
            }
        }
    }

    FE_CHECK_ARG(monomials_.size() == n, "BDMBasis modal basis size mismatch");
    FE_CHECK_ARG(std::all_of(monomials_.begin(), monomials_.end(),
                             [](const ModalPolynomial& poly) {
                                 return poly.num_terms == 1;
                             }),
                 "BDMBasis interior DOF assembly requires single-term modal polynomials");

    modal_power_limits_ = modal_power_limits(monomials_);
    const int max_px = modal_power_limits_[0];
    const int max_py = modal_power_limits_[1];
    const int max_pz = modal_power_limits_[2];
    std::vector<Real> A(n * n, Real(0));
    std::size_t row = 0;
    std::vector<Real> bvals;
    std::vector<Real> modal_dot(n, Real(0));
    std::vector<Vec3> test_values;
    std::vector<Real> power_x;
    std::vector<Real> power_y;
    std::vector<Real> power_z;

    if (type == ElementType::Triangle3) {
        const elements::ReferenceElement ref = elements::ReferenceElement::create(type);
        const LagrangeBasis edge_basis(ElementType::Line2, order_);
        const auto edge_quad = quadrature::QuadratureFactory::create(
            ElementType::Line2, 2 * order_ + 2, QuadratureType::GaussLegendre);

        for (std::size_t e = 0; e < ref.num_edges(); ++e) {
            const auto& edge_nodes = ref.edge_nodes(e);
            const Vec3 a = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(edge_nodes[0]));
            const Vec3 b = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(edge_nodes[1]));
            const Vec3 tvec = b - a;
            const Real len = tvec.norm();
            FE_CHECK_ARG(len > Real(0), "BDM triangle edge has zero length");
            const Vec3 t = tvec / len;
            const Vec3 nrm{t[1], -t[0], Real(0)};
            const Real J = len * Real(0.5);

            for (std::size_t q = 0; q < edge_quad->num_points(); ++q) {
                const Real s = edge_quad->point(q)[0];
                const Real tpar = (s + Real(1)) * Real(0.5);
                const Vec3 xi = a * (Real(1) - tpar) + b * tpar;
                bvals.clear();
                edge_basis.evaluate_values(Vec3{s, Real(0), Real(0)}, bvals);

                fill_powers(xi[0], max_px, power_x);
                fill_powers(xi[1], max_py, power_y);

                std::fill(modal_dot.begin(), modal_dot.end(), Real(0));
                for (std::size_t p = 0; p < n; ++p) {
                    const auto& poly = monomials_[p];
                    Real dot = Real(0);
                    for (int term = 0; term < poly.num_terms; ++term) {
                        const auto& mono = poly.terms[static_cast<std::size_t>(term)];
                        const Real mv = power_x[static_cast<std::size_t>(mono.px)] *
                                        power_y[static_cast<std::size_t>(mono.py)];
                        dot += mono.coefficient * nrm[static_cast<std::size_t>(mono.component)] * mv;
                    }
                    modal_dot[p] = dot;
                }

                const Real wt = edge_quad->weight(q) * J;
                for (std::size_t aidx = 0; aidx < bvals.size(); ++aidx) {
                    const Real wa = wt * bvals[aidx];
                    if (wa == Real(0)) {
                        continue;
                    }
                    const std::size_t r = row + aidx;
                    for (std::size_t p = 0; p < n; ++p) {
                        A[r * n + p] += wa * modal_dot[p];
                    }
                }
            }
            row += edge_basis.size();
        }

        if (order_ >= 2) {
            const auto interior_basis =
                cached_bdm_inner_nedelec_basis(ElementType::Triangle3, order_ - 2);
            const auto tri_quad = quadrature::QuadratureFactory::create(
                ElementType::Triangle3, 2 * order_ + 2, QuadratureType::GaussLegendre);

            for (std::size_t q = 0; q < tri_quad->num_points(); ++q) {
                const Vec3 xi = tri_quad->point(q);
                test_values.clear();
                interior_basis->evaluate_vector_values(xi, test_values);
                fill_powers(xi[0], max_px, power_x);
                fill_powers(xi[1], max_py, power_y);
                const Real wt = tri_quad->weight(q);

                std::fill(modal_dot.begin(), modal_dot.end(), Real(0));
                for (std::size_t p = 0; p < n; ++p) {
                    const auto& poly = monomials_[p];
                    Real dot = Real(0);
                    for (int term = 0; term < poly.num_terms; ++term) {
                        const auto& mono = poly.terms[static_cast<std::size_t>(term)];
                        const Real mv = power_x[static_cast<std::size_t>(mono.px)] *
                                        power_y[static_cast<std::size_t>(mono.py)];
                        dot += mono.coefficient * mv;
                    }
                    modal_dot[p] = dot;
                }

                for (std::size_t aidx = 0; aidx < test_values.size(); ++aidx) {
                    const Vec3& test = test_values[aidx];
                    const std::size_t r = row + aidx;
                    for (std::size_t p = 0; p < n; ++p) {
                        const auto& mono = monomials_[p].terms[0];
                        const Real weight = test[static_cast<std::size_t>(mono.component)];
                        A[r * n + p] += wt * modal_dot[p] * weight;
                    }
                }
            }
            row += interior_basis->size();
        }
    } else {
        const elements::ReferenceElement ref = elements::ReferenceElement::create(type);
        const LagrangeBasis face_basis(ElementType::Triangle3, order_);
        const auto tri_quad = quadrature::QuadratureFactory::create(
            ElementType::Triangle3, 2 * order_ + 2, QuadratureType::GaussLegendre);

        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const auto& face_nodes = ref.face_nodes(f);
            const Vec3 v0 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(face_nodes[0]));
            const Vec3 v1 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(face_nodes[1]));
            const Vec3 v2 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(face_nodes[2]));
            const Vec3 e01 = v1 - v0;
            const Vec3 e02 = v2 - v0;
            const Vec3 cross = cross3(e01, e02);
            const Real scale = cross.norm();
            FE_CHECK_ARG(scale > Real(0), "BDM tetra face has zero area");
            const Vec3 nrm = normalize3(cross);

            for (std::size_t q = 0; q < tri_quad->num_points(); ++q) {
                const auto uv = tri_quad->point(q);
                const Real u = uv[0];
                const Real v = uv[1];
                const Vec3 xi = v0 + e01 * u + e02 * v;
                bvals.clear();
                face_basis.evaluate_values(Vec3{u, v, Real(0)}, bvals);

                fill_powers(xi[0], max_px, power_x);
                fill_powers(xi[1], max_py, power_y);
                fill_powers(xi[2], max_pz, power_z);

                std::fill(modal_dot.begin(), modal_dot.end(), Real(0));
                for (std::size_t p = 0; p < n; ++p) {
                    const auto& poly = monomials_[p];
                    Real dot = Real(0);
                    for (int term = 0; term < poly.num_terms; ++term) {
                        const auto& mono = poly.terms[static_cast<std::size_t>(term)];
                        const Real mv = power_x[static_cast<std::size_t>(mono.px)] *
                                        power_y[static_cast<std::size_t>(mono.py)] *
                                        power_z[static_cast<std::size_t>(mono.pz)];
                        dot += mono.coefficient * nrm[static_cast<std::size_t>(mono.component)] * mv;
                    }
                    modal_dot[p] = dot;
                }

                const Real wt = tri_quad->weight(q) * scale;
                for (std::size_t aidx = 0; aidx < bvals.size(); ++aidx) {
                    const Real wa = wt * bvals[aidx];
                    if (wa == Real(0)) {
                        continue;
                    }
                    const std::size_t r = row + aidx;
                    for (std::size_t p = 0; p < n; ++p) {
                        A[r * n + p] += wa * modal_dot[p];
                    }
                }
            }

            row += face_basis.size();
        }

        if (order_ >= 2) {
            const auto interior_basis =
                cached_bdm_inner_nedelec_basis(ElementType::Tetra4, order_ - 2);
            const auto tet_quad = quadrature::QuadratureFactory::create(
                ElementType::Tetra4, 2 * order_ + 2, QuadratureType::GaussLegendre);

            for (std::size_t q = 0; q < tet_quad->num_points(); ++q) {
                const Vec3 xi = tet_quad->point(q);
                test_values.clear();
                interior_basis->evaluate_vector_values(xi, test_values);
                fill_powers(xi[0], max_px, power_x);
                fill_powers(xi[1], max_py, power_y);
                fill_powers(xi[2], max_pz, power_z);
                const Real wt = tet_quad->weight(q);

                for (std::size_t aidx = 0; aidx < test_values.size(); ++aidx) {
                    const Vec3& test = test_values[aidx];
                    const std::size_t r = row + aidx;
                    for (std::size_t p = 0; p < n; ++p) {
                        const auto& mono = monomials_[p].terms[0];
                        const Real mv = power_x[static_cast<std::size_t>(mono.px)] *
                                        power_y[static_cast<std::size_t>(mono.py)] *
                                        power_z[static_cast<std::size_t>(mono.pz)];
                        A[r * n + p] += wt * mono.coefficient * mv *
                                        test[static_cast<std::size_t>(mono.component)];
                    }
                }
            }
            row += interior_basis->size();
        }
    }

    FE_CHECK_ARG(row == n, "BDMBasis: DOF assembly did not fill matrix");
    const std::vector<Real> coeffs = invert_dense_matrix(std::move(A), n);
    modal_sparse_coeffs_ = build_sparse_modal_coefficients(coeffs, n, n);
    nodal_generated_ = true;
}

void BDMBasis::evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                      std::vector<math::Vector<Real, 3>>& values) const {
    if (element_type_ == ElementType::Triangle3 && !nodal_generated_) {
        const Real x = xi[0];
        const Real y = xi[1];
        values.resize(6);
        values[0] = math::Vector<Real, 3>{x, y - Real(1), Real(0)};
        values[1] = math::Vector<Real, 3>{Real(3) * x,
                                          Real(3) - Real(6) * x - Real(3) * y,
                                          Real(0)};
        values[2] = math::Vector<Real, 3>{x, y, Real(0)};
        values[3] = math::Vector<Real, 3>{-Real(3) * x, Real(3) * y, Real(0)};
        values[4] = math::Vector<Real, 3>{x - Real(1), y, Real(0)};
        values[5] = math::Vector<Real, 3>{Real(3) - Real(3) * x - Real(6) * y,
                                          Real(3) * y,
                                          Real(0)};
        return;
    }

    if (nodal_generated_) {
        evaluate_nodal_modal_vector_values_with_limits(
            monomials_, modal_sparse_coeffs_, size_, xi, modal_power_limits_, values);
        return;
    }

    const Real x = xi[0];
    const Real y = xi[1];

    // BDM1 on the reference quadrilateral [-1,1]^2:
    // two edge moments per edge (constant + linear), 8 basis functions total.
    values.resize(8);
    const Real half = Real(0.5);
    const Real one = Real(1);
    // Edge 0-1 (bottom, y=-1): outward normal -y
    values[0] = math::Vector<Real, 3>{Real(0), half * (y - one), Real(0)};                  // flux = 1
    values[1] = math::Vector<Real, 3>{Real(0), half * x * (y - one), Real(0)};              // flux = x
    // Edge 1-2 (right, x=+1): outward normal +x
    values[2] = math::Vector<Real, 3>{half * (one + x), Real(0), Real(0)};                  // flux = 1
    values[3] = math::Vector<Real, 3>{half * y * (one + x), Real(0), Real(0)};              // flux = y
    // Edge 2-3 (top, y=+1): outward normal +y
    values[4] = math::Vector<Real, 3>{Real(0), half * (one + y), Real(0)};                  // flux = 1
    values[5] = math::Vector<Real, 3>{Real(0), half * x * (one + y), Real(0)};              // flux = x
    // Edge 3-0 (left, x=-1): outward normal -x
    values[6] = math::Vector<Real, 3>{half * (x - one), Real(0), Real(0)};                  // flux = 1
    values[7] = math::Vector<Real, 3>{half * y * (x - one), Real(0), Real(0)};              // flux = y
}

void BDMBasis::evaluate_vector_jacobians(const math::Vector<Real, 3>& xi,
                                         std::vector<VectorJacobian>& jacobians) const {
    if (element_type_ == ElementType::Triangle3 && !nodal_generated_) {
        jacobians.assign(6, VectorJacobian{});
        jacobians[0](0, 0) = Real(1);
        jacobians[0](1, 1) = Real(1);
        jacobians[1](0, 0) = Real(3);
        jacobians[1](1, 0) = -Real(6);
        jacobians[1](1, 1) = -Real(3);
        jacobians[2](0, 0) = Real(1);
        jacobians[2](1, 1) = Real(1);
        jacobians[3](0, 0) = -Real(3);
        jacobians[3](1, 1) = Real(3);
        jacobians[4](0, 0) = Real(1);
        jacobians[4](1, 1) = Real(1);
        jacobians[5](0, 0) = -Real(3);
        jacobians[5](0, 1) = -Real(6);
        jacobians[5](1, 1) = Real(3);
        return;
    }

    if (nodal_generated_) {
        evaluate_nodal_modal_vector_jacobians_with_limits(
            monomials_, modal_sparse_coeffs_, size_, xi, modal_power_limits_, jacobians);
        return;
    }

    const Real x = xi[0];
    const Real y = xi[1];
    const Real half = Real(0.5);
    const Real one = Real(1);

    jacobians.assign(8, VectorJacobian{});
    jacobians[0](1, 1) = half;
    jacobians[1](1, 0) = half * (y - one);
    jacobians[1](1, 1) = half * x;
    jacobians[2](0, 0) = half;
    jacobians[3](0, 0) = half * y;
    jacobians[3](0, 1) = half * (one + x);
    jacobians[4](1, 1) = half;
    jacobians[5](1, 0) = half * (one + y);
    jacobians[5](1, 1) = half * x;
    jacobians[6](0, 0) = half;
    jacobians[7](0, 0) = half * y;
    jacobians[7](0, 1) = half * (x - one);
}

void BDMBasis::evaluate_divergence(const math::Vector<Real, 3>& xi,
                                   std::vector<Real>& divergence) const {
    if (element_type_ == ElementType::Triangle3 && !nodal_generated_) {
        (void)xi;
        divergence.resize(6);
        divergence[0] = Real(2);
        divergence[1] = Real(0);
        divergence[2] = Real(2);
        divergence[3] = Real(0);
        divergence[4] = Real(2);
        divergence[5] = Real(0);
        return;
    }

    if (nodal_generated_) {
        evaluate_nodal_modal_divergence_with_limits(
            monomials_, modal_sparse_coeffs_, size_, xi, modal_power_limits_, divergence);
        return;
    }

    const Real x = xi[0];
    const Real y = xi[1];

    divergence.resize(8);
    const Real half = Real(0.5);
    divergence[0] = half;
    divergence[1] = half * x;
    divergence[2] = half;
    divergence[3] = half * y;
    divergence[4] = half;
    divergence[5] = half * x;
    divergence[6] = half;
    divergence[7] = half * y;
}

void BDMBasis::evaluate_vector_at_quadrature_points_strided(
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
            "BDMBasis");
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
                                       "BDMBasis");
}

} // namespace basis
} // namespace FE
} // namespace svmp
