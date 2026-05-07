/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "LagrangeBasis.h"
#include "NodeOrderingConventions.h"
#include "detail/LagrangeBasisPyramidDetail.h"
#include "detail/LagrangeBasisSimplexDetail.h"
#include "detail/LagrangeBasisUtilityDetail.h"
#include <algorithm>
#include <cmath>
#include <map>

namespace svmp {
namespace FE {
namespace basis {

namespace {

enum class LagrangeTopology {
    Point,
    Line,
    Triangle,
    Quadrilateral,
    Tetrahedron,
    Hexahedron,
    Wedge,
    Pyramid,
};

struct LagrangeTopologyTraits {
    LagrangeTopology topology;
    int dimension;
};

constexpr Real kNodeCoordTolerance = Real(1e-12);

LagrangeTopologyTraits lagrange_topology_traits(ElementType type) {
    switch (type) {
        case ElementType::Point1:
            return {LagrangeTopology::Point, 0};
        case ElementType::Line2:
        case ElementType::Line3:
            return {LagrangeTopology::Line, 1};
        case ElementType::Triangle3:
        case ElementType::Triangle6:
            return {LagrangeTopology::Triangle, 2};
        case ElementType::Quad4:
        case ElementType::Quad8:
        case ElementType::Quad9:
            return {LagrangeTopology::Quadrilateral, 2};
        case ElementType::Tetra4:
        case ElementType::Tetra10:
            return {LagrangeTopology::Tetrahedron, 3};
        case ElementType::Hex8:
        case ElementType::Hex20:
        case ElementType::Hex27:
            return {LagrangeTopology::Hexahedron, 3};
        case ElementType::Wedge6:
        case ElementType::Wedge15:
        case ElementType::Wedge18:
            return {LagrangeTopology::Wedge, 3};
        case ElementType::Pyramid5:
        case ElementType::Pyramid13:
        case ElementType::Pyramid14:
            return {LagrangeTopology::Pyramid, 3};
    }

    throw BasisElementCompatibilityException("Unsupported element type for LagrangeBasis",
                                             __FILE__, __LINE__, __func__);
}

std::size_t lattice_index_pm_one(Real coord, int order, const char* context) {
    if (order <= 0) {
        if (std::abs(coord) > kNodeCoordTolerance) {
            throw BasisNodeOrderingException(context, __FILE__, __LINE__, __func__);
        }
        return 0;
    }

    const Real scaled = (coord + Real(1)) * static_cast<Real>(order) / Real(2);
    const long idx = std::lround(scaled);
    if (idx < 0 || idx > order ||
        std::abs(coord - detail::equispaced_pm_one_coord(static_cast<int>(idx), order)) > kNodeCoordTolerance) {
        throw BasisNodeOrderingException(context, __FILE__, __LINE__, __func__);
    }
    return static_cast<std::size_t>(idx);
}

int simplex_lattice_index(Real coord, int order, const char* context) {
    if (order <= 0) {
        if (std::abs(coord - Real(0)) > kNodeCoordTolerance &&
            std::abs(coord - Real(0.25)) > kNodeCoordTolerance &&
            std::abs(coord - Real(1) / Real(3)) > kNodeCoordTolerance) {
            throw BasisNodeOrderingException(context, __FILE__, __LINE__, __func__);
        }
        return 0;
    }

    const Real scaled = coord * static_cast<Real>(order);
    const long idx = std::lround(scaled);
    const Real reconstructed = static_cast<Real>(idx) / static_cast<Real>(order);
    if (idx < 0 || idx > order || std::abs(coord - reconstructed) > kNodeCoordTolerance) {
        throw BasisNodeOrderingException(context, __FILE__, __LINE__, __func__);
    }
    return static_cast<int>(idx);
}

std::array<int, 4> triangle_exponents_from_public_node(const math::Vector<Real, 3>& node,
                                                       int order) {
    if (order == 0) {
        return {0, 0, 0, 0};
    }

    const int j = simplex_lattice_index(node[0], order,
                                        "LagrangeBasis: invalid triangle node coordinate for public ordering");
    const int k = simplex_lattice_index(node[1], order,
                                        "LagrangeBasis: invalid triangle node coordinate for public ordering");
    const int i = order - j - k;
    if (i < 0) {
        throw BasisNodeOrderingException("LagrangeBasis: invalid triangle barycentric coordinates for public ordering",
                                         __FILE__, __LINE__, __func__);
    }
    return {i, j, k, 0};
}

std::array<int, 4> tetrahedron_exponents_from_public_node(const math::Vector<Real, 3>& node,
                                                          int order) {
    if (order == 0) {
        return {0, 0, 0, 0};
    }

    const int j = simplex_lattice_index(node[0], order,
                                        "LagrangeBasis: invalid tetrahedron node x-coordinate for public ordering");
    const int k = simplex_lattice_index(node[1], order,
                                        "LagrangeBasis: invalid tetrahedron node y-coordinate for public ordering");
    const int l = simplex_lattice_index(node[2], order,
                                        "LagrangeBasis: invalid tetrahedron node z-coordinate for public ordering");
    const int i = order - j - k - l;
    if (i < 0) {
        throw BasisNodeOrderingException("LagrangeBasis: invalid tetrahedron barycentric coordinates for public ordering",
                                         __FILE__, __LINE__, __func__);
    }
    return {i, j, k, l};
}

struct NormalizedLagrangeRequest {
    ElementType element_type;
    int order;
};

// Non-owning view of the per-axis 1D Lagrange basis evaluations
// (values, first derivative, second derivative), each of length `size`.
struct AxisBasisEvaluations {
    const Real* values;
    const Real* first;
    const Real* second;
    std::size_t size;
};

AxisBasisEvaluations constant_axis_basis() {
    static const Real kOne[1]  = {Real(1)};
    static const Real kZero[1] = {Real(0)};
    return AxisBasisEvaluations{kOne, kZero, kZero, 1};
}

// C1+C6: Horner-form evaluator for the precomputed 1D Lagrange basis.
//
// Inputs are precomputed monomial coefficients of L_i(x), L_i'(x), L_i''(x)
// (built once at LagrangeBasis construction). Evaluation is purely
// multiply-add on the coefficients — no divisions and no node-position
// lookups in the hot path. Templated on N for compile-time loop unrolling
// and FMA-friendly straight-line code on the common Hex/Quad/Line orders.
//
// Layout:
//   v_coeffs:  N * N entries; row i holds [c_i0, c_i1, ..., c_i(N-1)]
//              such that L_i(x) = sum_k c_ik * x^k
//   d_coeffs:  N * (N-1) entries; row i holds derivative coefficients of L_i'(x)
//   d2_coeffs: N * (N-2) entries; row i holds coefficients of L_i''(x)
//              (only valid when N >= 3)
template<int N>
inline void evaluate_1d_horner_impl(const Real* v_coeffs,
                                    const Real* d_coeffs,
                                    const Real* d2_coeffs,
                                    Real xi,
                                    Real* values, Real* first, Real* second) {
    if constexpr (N == 1) {
        values[0] = v_coeffs[0];
        if (first)  first[0]  = Real(0);
        if (second) second[0] = Real(0);
        return;
    } else {
        // Values: degree N-1 polynomials.
        for (int i = 0; i < N; ++i) {
            const Real* c = v_coeffs + i * N;
            Real r = c[N - 1];
            for (int k = N - 1; k > 0; --k) {
                r = r * xi + c[k - 1];
            }
            values[i] = r;
        }

        if (!first && !second) return;

        if (first) {
            // First derivatives: degree N-2 polynomials (per row of d_coeffs).
            for (int i = 0; i < N; ++i) {
                const Real* c = d_coeffs + i * (N - 1);
                Real r = c[N - 2];
                for (int k = N - 2; k > 0; --k) {
                    r = r * xi + c[k - 1];
                }
                first[i] = r;
            }
        }

        if (!second) return;

        if constexpr (N <= 2) {
            for (int i = 0; i < N; ++i) second[i] = Real(0);
        } else {
            // Second derivatives: degree N-3 polynomials (per row of d2_coeffs).
            for (int i = 0; i < N; ++i) {
                const Real* c = d2_coeffs + i * (N - 2);
                Real r = c[N - 3];
                for (int k = N - 3; k > 0; --k) {
                    r = r * xi + c[k - 1];
                }
                second[i] = r;
            }
        }
    }
}

// Runtime fallback for orders >= 5 (axis size N > 5). Same Horner kernel
// but with N as a parameter rather than a template.
void evaluate_1d_horner_runtime(const Real* v_coeffs,
                                const Real* d_coeffs,
                                const Real* d2_coeffs,
                                int n_axis, Real xi,
                                Real* values, Real* first, Real* second) {
    const int N = n_axis;
    if (N == 1) {
        values[0] = v_coeffs[0];
        if (first)  first[0]  = Real(0);
        if (second) second[0] = Real(0);
        return;
    }

    for (int i = 0; i < N; ++i) {
        const Real* c = v_coeffs + i * N;
        Real r = c[N - 1];
        for (int k = N - 1; k > 0; --k) {
            r = r * xi + c[k - 1];
        }
        values[i] = r;
    }

    if (!first && !second) return;

    if (first) {
        for (int i = 0; i < N; ++i) {
            const Real* c = d_coeffs + i * (N - 1);
            Real r = c[N - 2];
            for (int k = N - 2; k > 0; --k) {
                r = r * xi + c[k - 1];
            }
            first[i] = r;
        }
    }

    if (!second) return;

    if (N <= 2) {
        for (int i = 0; i < N; ++i) second[i] = Real(0);
    } else {
        for (int i = 0; i < N; ++i) {
            const Real* c = d2_coeffs + i * (N - 2);
            Real r = c[N - 3];
            for (int k = N - 3; k > 0; --k) {
                r = r * xi + c[k - 1];
            }
            second[i] = r;
        }
    }
}

// 1D Lagrange-basis evaluator. Writes n_axis entries to each non-null output
// buffer without allocating or dividing. Dispatches to compile-time-N
// specializations for sizes 1..5 (orders 0..4 — the common cases) and falls
// through to a runtime variant for higher orders.
void evaluate_1d_basis_to(const Real* v_coeffs,
                          const Real* d_coeffs,
                          const Real* d2_coeffs,
                          int n_axis, Real xi,
                          Real* values, Real* first, Real* second) {
    switch (n_axis) {
        case 1: evaluate_1d_horner_impl<1>(v_coeffs, d_coeffs, d2_coeffs, xi, values, first, second); return;
        case 2: evaluate_1d_horner_impl<2>(v_coeffs, d_coeffs, d2_coeffs, xi, values, first, second); return;
        case 3: evaluate_1d_horner_impl<3>(v_coeffs, d_coeffs, d2_coeffs, xi, values, first, second); return;
        case 4: evaluate_1d_horner_impl<4>(v_coeffs, d_coeffs, d2_coeffs, xi, values, first, second); return;
        case 5: evaluate_1d_horner_impl<5>(v_coeffs, d_coeffs, d2_coeffs, xi, values, first, second); return;
        default:
            evaluate_1d_horner_runtime(v_coeffs, d_coeffs, d2_coeffs, n_axis, xi, values, first, second);
            return;
    }
}

// Per-axis storage (values, first derivative, second derivative). Backed by
// thread_local std::vector that grows lazily; subsequent calls reuse capacity
// with no reallocation.
struct AxisScratch {
    std::vector<Real> values;
    std::vector<Real> first;
    std::vector<Real> second;

    void reserveFor(std::size_t n) {
        if (values.size() < n) values.resize(n);
        if (first.size() < n) first.resize(n);
        if (second.size() < n) second.resize(n);
    }
};

// Caller-provided scratch buffers used by tensor-product evaluation. Three
// independent axes plus reusable simplex/wedge intermediates.
struct LagrangeEvaluateScratch {
    AxisScratch axis_x;
    AxisScratch axis_y;
    AxisScratch axis_z;

    std::vector<Real> tri_values;
    std::vector<Gradient> tri_gradients;
    std::vector<Hessian> tri_hessians;
};

LagrangeEvaluateScratch& evaluate_scratch() {
    thread_local LagrangeEvaluateScratch s;
    return s;
}

// Selects which derivative passes are computed by the 1D evaluator.
enum class AxisDeriv {
    ValuesOnly,           // skip first and second
    ValuesAndFirst,       // for gradients
    ValuesAndFirstAndSecond, // for hessians or fused evaluate_all
};

// Fill axis scratch and return a non-owning view. Uncomputed slots still have
// valid pointers to scratch storage (they may hold stale data) — callers must
// only read the slots they requested via `level`. Uses precomputed Horner
// coefficients (C1+C6); no divisions in the hot path.
AxisBasisEvaluations fill_axis_scratch(AxisScratch& s,
                                       const Real* v_coeffs,
                                       const Real* d_coeffs,
                                       const Real* d2_coeffs,
                                       int n_axis, Real xi,
                                       AxisDeriv level) {
    const std::size_t n = static_cast<std::size_t>(n_axis);
    s.reserveFor(n);
    Real* first  = (level == AxisDeriv::ValuesOnly) ? nullptr : s.first.data();
    Real* second = (level == AxisDeriv::ValuesAndFirstAndSecond) ? s.second.data() : nullptr;
    evaluate_1d_basis_to(v_coeffs, d_coeffs, d2_coeffs, n_axis, xi, s.values.data(), first, second);
    return AxisBasisEvaluations{s.values.data(), s.first.data(), s.second.data(), n};
}

// Maximum yz-table footprint that fits comfortably on the stack for typical
// element orders. Up to order 7 per axis (8x8 yz pairs = 64 entries per table).
// Higher orders fall back to thread_local heap buffers.
inline constexpr std::size_t kMaxStackYZ = 64;

// Fused sum-factorized tensor-product evaluator (B1 + B4).
//
// Precomputes one to six (ny x nz)-shaped tables of partial products
// `M_xy[j*nz + k]` so that the inner per-node loop performs at most one
// multiplication per output instead of two. With all three output buffers
// supplied, this is the fused values + gradients + hessians path that shares
// every per-axis evaluation.
//
// Per-node multiply count (vs. the unfactored variants):
//   values only       : 1  (was 2)
//   gradients only    : 3  (was 6)
//   hessians only     : 6  (was 12)
//   all three (B4)    : 10 (was 20)
//
// Dimensional scope: works uniformly for Line/Quadrilateral/Hexahedron with
// the unused axes' size folded to 1 via constant_axis_basis().
void evaluate_tensor_product_factorized(
    const std::vector<std::array<std::size_t, 3>>& tensor_indices,
    const AxisBasisEvaluations& x_axis,
    const AxisBasisEvaluations& y_axis,
    const AxisBasisEvaluations& z_axis,
    std::vector<Real>* values_out,
    std::vector<Gradient>* gradients_out,
    std::vector<Hessian>* hessians_out) {
    const std::size_t ny = y_axis.size;
    const std::size_t nz = z_axis.size;
    const std::size_t nyz = ny * nz;
    const bool need_grad = (gradients_out != nullptr);
    const bool need_hess = (hessians_out != nullptr);

    Real Mvv_stack[kMaxStackYZ];
    Real Mdv_stack[kMaxStackYZ];
    Real Mvd_stack[kMaxStackYZ];
    Real Md2v_stack[kMaxStackYZ];
    Real Mvd2_stack[kMaxStackYZ];
    Real Mdd_stack[kMaxStackYZ];

    thread_local std::vector<Real> Mvv_heap, Mdv_heap, Mvd_heap;
    thread_local std::vector<Real> Md2v_heap, Mvd2_heap, Mdd_heap;

    Real* Mvv;
    Real* Mdv;
    Real* Mvd;
    Real* Md2v;
    Real* Mvd2;
    Real* Mdd;
    if (nyz <= kMaxStackYZ) {
        Mvv = Mvv_stack;
        Mdv = Mdv_stack;
        Mvd = Mvd_stack;
        Md2v = Md2v_stack;
        Mvd2 = Mvd2_stack;
        Mdd = Mdd_stack;
    } else {
        if (Mvv_heap.size() < nyz) {
            Mvv_heap.resize(nyz);
            Mdv_heap.resize(nyz);
            Mvd_heap.resize(nyz);
            Md2v_heap.resize(nyz);
            Mvd2_heap.resize(nyz);
            Mdd_heap.resize(nyz);
        }
        Mvv = Mvv_heap.data();
        Mdv = Mdv_heap.data();
        Mvd = Mvd_heap.data();
        Md2v = Md2v_heap.data();
        Mvd2 = Mvd2_heap.data();
        Mdd = Mdd_heap.data();
    }

    // M_vv is required by every output (values, ∂ξ, ∂ξ²).
    for (std::size_t j = 0; j < ny; ++j) {
        const Real yv = y_axis.values[j];
        for (std::size_t k = 0; k < nz; ++k) {
            Mvv[j * nz + k] = yv * z_axis.values[k];
        }
    }

    if (need_grad || need_hess) {
        for (std::size_t j = 0; j < ny; ++j) {
            const Real yv = y_axis.values[j];
            const Real yd = y_axis.first[j];
            for (std::size_t k = 0; k < nz; ++k) {
                Mdv[j * nz + k] = yd * z_axis.values[k];
                Mvd[j * nz + k] = yv * z_axis.first[k];
            }
        }
    }

    if (need_hess) {
        for (std::size_t j = 0; j < ny; ++j) {
            const Real yv = y_axis.values[j];
            const Real yd = y_axis.first[j];
            const Real yd2 = y_axis.second[j];
            for (std::size_t k = 0; k < nz; ++k) {
                Md2v[j * nz + k] = yd2 * z_axis.values[k];
                Mvd2[j * nz + k] = yv  * z_axis.second[k];
                Mdd[j * nz + k]  = yd  * z_axis.first[k];
            }
        }
    }

    const std::size_t n_nodes = tensor_indices.size();
    if (values_out)    values_out->resize(n_nodes);
    if (gradients_out) gradients_out->resize(n_nodes);
    if (hessians_out)  hessians_out->resize(n_nodes);

    for (std::size_t n = 0; n < n_nodes; ++n) {
        const auto& idx = tensor_indices[n];
        const std::size_t i = idx[0];
        const std::size_t jk = idx[1] * nz + idx[2];

        const Real Lx = x_axis.values[i];

        if (values_out) {
            (*values_out)[n] = Lx * Mvv[jk];
        }

        if (need_grad) {
            const Real dLx = x_axis.first[i];
            auto& g = (*gradients_out)[n];
            g[0] = dLx * Mvv[jk];
            g[1] = Lx  * Mdv[jk];
            g[2] = Lx  * Mvd[jk];
        }

        if (need_hess) {
            const Real dLx  = x_axis.first[i];
            const Real d2Lx = x_axis.second[i];
            Hessian H{};
            H(0, 0) = d2Lx * Mvv[jk];
            H(1, 1) = Lx   * Md2v[jk];
            H(2, 2) = Lx   * Mvd2[jk];
            H(0, 1) = dLx  * Mdv[jk];   H(1, 0) = H(0, 1);
            H(0, 2) = dLx  * Mvd[jk];   H(2, 0) = H(0, 2);
            H(1, 2) = Lx   * Mdd[jk];   H(2, 1) = H(1, 2);
            (*hessians_out)[n] = H;
        }
    }
}

// Single-output convenience wrappers — thin pass-throughs onto the fused
// factorized core so all paths share the same sum-factorization kernel.
inline void evaluate_tensor_product_values(
    const std::vector<std::array<std::size_t, 3>>& tensor_indices,
    const AxisBasisEvaluations& x_axis,
    const AxisBasisEvaluations& y_axis,
    const AxisBasisEvaluations& z_axis,
    std::vector<Real>& values) {
    evaluate_tensor_product_factorized(tensor_indices, x_axis, y_axis, z_axis,
                                       &values, nullptr, nullptr);
}

inline void evaluate_tensor_product_gradients(
    const std::vector<std::array<std::size_t, 3>>& tensor_indices,
    const AxisBasisEvaluations& x_axis,
    const AxisBasisEvaluations& y_axis,
    const AxisBasisEvaluations& z_axis,
    std::vector<Gradient>& gradients) {
    evaluate_tensor_product_factorized(tensor_indices, x_axis, y_axis, z_axis,
                                       nullptr, &gradients, nullptr);
}

inline void evaluate_tensor_product_hessians(
    const std::vector<std::array<std::size_t, 3>>& tensor_indices,
    const AxisBasisEvaluations& x_axis,
    const AxisBasisEvaluations& y_axis,
    const AxisBasisEvaluations& z_axis,
    std::vector<Hessian>& hessians) {
    evaluate_tensor_product_factorized(tensor_indices, x_axis, y_axis, z_axis,
                                       nullptr, nullptr, &hessians);
}

// E1: hardcoded TRI6 (order-2 triangle) fast path. Inlines the recursive
// simplex_lagrange_factor_sequence for p=2 and writes outputs directly.
//
// For p=2, the factor sequence is:
//   phi[0] = 1, phi[1] = 2λ, phi[2] = λ(2λ - 1)
//   dphi[0] = 0, dphi[1] = 2, dphi[2] = 4λ - 1
//   d2phi[0] = 0, d2phi[1] = 0, d2phi[2] = 4
inline void evaluate_tri6_p2_fast(
    const math::Vector<Real, 3>& xi,
    const std::vector<std::array<int, 4>>& simplex_exponents,
    std::vector<Real>* values_out,
    std::vector<Gradient>* gradients_out,
    std::vector<Hessian>* hessians_out) {
    const Real l1 = xi[0];
    const Real l2 = xi[1];
    const Real l0 = Real(1) - l1 - l2;

    const Real phi_a[3] = {Real(1), Real(2) * l0, l0 * (Real(2) * l0 - Real(1))};
    const Real phi_b[3] = {Real(1), Real(2) * l1, l1 * (Real(2) * l1 - Real(1))};
    const Real phi_c[3] = {Real(1), Real(2) * l2, l2 * (Real(2) * l2 - Real(1))};

    const bool need_grad = (gradients_out != nullptr);
    const bool need_hess = (hessians_out != nullptr);

    const Real dphi_a[3] = {Real(0), Real(2), Real(4) * l0 - Real(1)};
    const Real dphi_b[3] = {Real(0), Real(2), Real(4) * l1 - Real(1)};
    const Real dphi_c[3] = {Real(0), Real(2), Real(4) * l2 - Real(1)};
    constexpr Real d2_p2[3] = {Real(0), Real(0), Real(4)};

    const std::size_t n_nodes = simplex_exponents.size();
    if (values_out)    values_out->resize(n_nodes);
    if (gradients_out) gradients_out->resize(n_nodes);
    if (hessians_out)  hessians_out->resize(n_nodes);

    for (std::size_t n = 0; n < n_nodes; ++n) {
        const auto& e = simplex_exponents[n];
        const std::size_t i = static_cast<std::size_t>(e[0]);
        const std::size_t j = static_cast<std::size_t>(e[1]);
        const std::size_t k = static_cast<std::size_t>(e[2]);
        const Real va = phi_a[i];
        const Real vb = phi_b[j];
        const Real vc = phi_c[k];

        if (values_out) {
            (*values_out)[n] = va * vb * vc;
        }

        if (need_grad || need_hess) {
            const Real Da = dphi_a[i];
            const Real Db = dphi_b[j];
            const Real Dc = dphi_c[k];

            if (need_grad) {
                const Real dl_a = Da * vb * vc;
                const Real dl_b = va * Db * vc;
                const Real dl_c = va * vb * Dc;
                Gradient& g = (*gradients_out)[n];
                g[0] = dl_b - dl_a;
                g[1] = dl_c - dl_a;
                g[2] = Real(0);
            }

            if (need_hess) {
                const Real DDa = d2_p2[i];
                const Real DDb = d2_p2[j];
                const Real DDc = d2_p2[k];
                const Real H00 = DDa * vb * vc;
                const Real H11 = va * DDb * vc;
                const Real H22 = va * vb * DDc;
                const Real H01 = Da * Db * vc;
                const Real H02 = Da * vb * Dc;
                const Real H12 = va * Db * Dc;
                Hessian H{};
                H(0, 0) = H00 - Real(2) * H01 + H11;
                H(1, 1) = H00 - Real(2) * H02 + H22;
                H(0, 1) = H00 - H01 - H02 + H12;
                H(1, 0) = H(0, 1);
                (*hessians_out)[n] = H;
            }
        }
    }
}

// E1: hardcoded TET10 (order-2 tet) fast path.
inline void evaluate_tet10_p2_fast(
    const math::Vector<Real, 3>& xi,
    const std::vector<std::array<int, 4>>& simplex_exponents,
    std::vector<Real>* values_out,
    std::vector<Gradient>* gradients_out,
    std::vector<Hessian>* hessians_out) {
    const Real l1 = xi[0];
    const Real l2 = xi[1];
    const Real l3 = xi[2];
    const Real l0 = Real(1) - l1 - l2 - l3;

    const Real phi_a[3] = {Real(1), Real(2) * l0, l0 * (Real(2) * l0 - Real(1))};
    const Real phi_b[3] = {Real(1), Real(2) * l1, l1 * (Real(2) * l1 - Real(1))};
    const Real phi_c[3] = {Real(1), Real(2) * l2, l2 * (Real(2) * l2 - Real(1))};
    const Real phi_d[3] = {Real(1), Real(2) * l3, l3 * (Real(2) * l3 - Real(1))};

    const bool need_grad = (gradients_out != nullptr);
    const bool need_hess = (hessians_out != nullptr);

    const Real dphi_a[3] = {Real(0), Real(2), Real(4) * l0 - Real(1)};
    const Real dphi_b[3] = {Real(0), Real(2), Real(4) * l1 - Real(1)};
    const Real dphi_c[3] = {Real(0), Real(2), Real(4) * l2 - Real(1)};
    const Real dphi_d[3] = {Real(0), Real(2), Real(4) * l3 - Real(1)};
    constexpr Real d2_p2[3] = {Real(0), Real(0), Real(4)};

    const std::size_t n_nodes = simplex_exponents.size();
    if (values_out)    values_out->resize(n_nodes);
    if (gradients_out) gradients_out->resize(n_nodes);
    if (hessians_out)  hessians_out->resize(n_nodes);

    for (std::size_t n = 0; n < n_nodes; ++n) {
        const auto& e = simplex_exponents[n];
        const std::size_t i = static_cast<std::size_t>(e[0]);
        const std::size_t j = static_cast<std::size_t>(e[1]);
        const std::size_t k = static_cast<std::size_t>(e[2]);
        const std::size_t l = static_cast<std::size_t>(e[3]);
        const Real va = phi_a[i];
        const Real vb = phi_b[j];
        const Real vc = phi_c[k];
        const Real vd = phi_d[l];

        if (values_out) {
            (*values_out)[n] = va * vb * vc * vd;
        }

        if (need_grad || need_hess) {
            const Real Da = dphi_a[i];
            const Real Db = dphi_b[j];
            const Real Dc = dphi_c[k];
            const Real Dd = dphi_d[l];

            if (need_grad) {
                const Real dl_a = Da * vb * vc * vd;
                const Real dl_b = va * Db * vc * vd;
                const Real dl_c = va * vb * Dc * vd;
                const Real dl_d = va * vb * vc * Dd;
                Gradient& g = (*gradients_out)[n];
                g[0] = dl_b - dl_a;
                g[1] = dl_c - dl_a;
                g[2] = dl_d - dl_a;
            }

            if (need_hess) {
                const Real DDa = d2_p2[i];
                const Real DDb = d2_p2[j];
                const Real DDc = d2_p2[k];
                const Real DDd = d2_p2[l];
                const Real H00 = DDa * vb * vc * vd;
                const Real H11 = va * DDb * vc * vd;
                const Real H22 = va * vb * DDc * vd;
                const Real H33 = va * vb * vc * DDd;
                const Real H01 = Da * Db * vc * vd;
                const Real H02 = Da * vb * Dc * vd;
                const Real H03 = Da * vb * vc * Dd;
                const Real H12 = va * Db * Dc * vd;
                const Real H13 = va * Db * vc * Dd;
                const Real H23 = va * vb * Dc * Dd;
                Hessian H{};
                H(0, 0) = H00 - Real(2) * H01 + H11;
                H(1, 1) = H00 - Real(2) * H02 + H22;
                H(2, 2) = H00 - Real(2) * H03 + H33;
                H(0, 1) = H00 - H01 - H02 + H12;
                H(1, 0) = H(0, 1);
                H(0, 2) = H00 - H01 - H03 + H13;
                H(2, 0) = H(0, 2);
                H(1, 2) = H00 - H02 - H03 + H23;
                H(2, 1) = H(1, 2);
                (*hessians_out)[n] = H;
            }
        }
    }
}

NormalizedLagrangeRequest normalize_lagrange_request(ElementType element_type, int order) {
    switch (element_type) {
        case ElementType::Line3:
            return {ElementType::Line2, std::max(order, 2)};
        case ElementType::Triangle6:
            return {ElementType::Triangle3, std::max(order, 2)};
        case ElementType::Quad9:
            return {ElementType::Quad4, std::max(order, 2)};
        case ElementType::Quad8:
            throw NotImplementedException("Quad8 serendipity Lagrange basis not implemented",
                                          __FILE__, __LINE__, __func__);
        case ElementType::Tetra10:
            return {ElementType::Tetra4, std::max(order, 2)};
        case ElementType::Hex27:
            return {ElementType::Hex8, std::max(order, 2)};
        case ElementType::Hex20:
            throw NotImplementedException("Hex20 serendipity Lagrange basis not implemented",
                                          __FILE__, __LINE__, __func__);
        case ElementType::Wedge18:
            return {ElementType::Wedge6, std::max(order, 2)};
        case ElementType::Wedge15:
            throw NotImplementedException("Wedge15 serendipity Lagrange basis not implemented",
                                          __FILE__, __LINE__, __func__);
        case ElementType::Pyramid13:
            throw NotImplementedException(
                "Pyramid13 is a serendipity variant; use SerendipityBasis (Pyramid13) or the complete-family Lagrange path via LagrangeBasis (Pyramid5, order >= 2)",
                __FILE__, __LINE__, __func__);
        case ElementType::Pyramid14:
            return {ElementType::Pyramid5, std::max(order, 2)};
        default:
            return {element_type, order};
    }
}

} // namespace

LagrangeBasis::LagrangeBasis(ElementType type, int order)
    : element_type_(type), dimension_(0), order_(order) {
    const NormalizedLagrangeRequest normalized = normalize_lagrange_request(element_type_, order_);
    element_type_ = normalized.element_type;
    order_ = normalized.order;

    if (order_ < 0) {
        throw BasisConfigurationException("LagrangeBasis requires non-negative polynomial order",
                                          __FILE__, __LINE__, __func__);
    }

    dimension_ = lagrange_topology_traits(element_type_).dimension;

    init_nodes();
}

void LagrangeBasis::init_nodes() {
    nodes_.clear();
    nodes_1d_.clear();
    tensor_indices_.clear();
    simplex_exponents_.clear();
    wedge_indices_.clear();
    axis_v_coeffs_.clear();
    axis_d_coeffs_.clear();
    axis_d2_coeffs_.clear();
    const auto topology = lagrange_topology_traits(element_type_).topology;
    topology_id_ = static_cast<int>(topology);  // C5: cache topology
    switch (topology) {
        case LagrangeTopology::Point:
            build_point_nodes();
            return;
        case LagrangeTopology::Line:
            build_tensor_product_nodes(1);
            compute_axis_monomial_coefficients();
            return;
        case LagrangeTopology::Quadrilateral:
            build_tensor_product_nodes(2);
            compute_axis_monomial_coefficients();
            return;
        case LagrangeTopology::Hexahedron:
            build_tensor_product_nodes(3);
            compute_axis_monomial_coefficients();
            return;
        case LagrangeTopology::Triangle:
        case LagrangeTopology::Tetrahedron:
            build_simplex_nodes();
            return;
        case LagrangeTopology::Wedge:
            build_wedge_nodes();
            compute_axis_monomial_coefficients();
            return;
        case LagrangeTopology::Pyramid:
            build_pyramid_nodes();
            return;
    }

    throw BasisElementCompatibilityException("Unsupported element type in LagrangeBasis::init_nodes",
                                             __FILE__, __LINE__, __func__);
}

void LagrangeBasis::compute_axis_monomial_coefficients() {
    const int N = static_cast<int>(nodes_1d_.size());
    if (N == 0) return;

    axis_v_coeffs_.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(N), Real(0));
    if (N >= 2) {
        axis_d_coeffs_.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(N - 1), Real(0));
    }
    if (N >= 3) {
        axis_d2_coeffs_.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(N - 2), Real(0));
    }

    if (N == 1) {
        axis_v_coeffs_[0] = Real(1);
        return;
    }

    // For each L_i, compute monomial coefficients of P_i(x) = prod_{j != i} (x - x_j),
    // then divide by w_i = prod_{j != i} (x_i - x_j).
    std::vector<Real> coeffs;
    coeffs.reserve(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        coeffs.assign(1, Real(1));  // start with constant polynomial 1
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            // Multiply (x - x_j) into coeffs (in-place via temp).
            std::vector<Real> next(coeffs.size() + 1, Real(0));
            for (std::size_t k = 0; k < coeffs.size(); ++k) {
                next[k]     -= nodes_1d_[static_cast<std::size_t>(j)] * coeffs[k];
                next[k + 1] += coeffs[k];
            }
            coeffs.swap(next);
        }
        // Divide by w_i.
        Real denom = Real(1);
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            denom *= (nodes_1d_[static_cast<std::size_t>(i)] - nodes_1d_[static_cast<std::size_t>(j)]);
        }
        const Real inv_denom = Real(1) / denom;
        for (int k = 0; k < N; ++k) {
            axis_v_coeffs_[static_cast<std::size_t>(i) * static_cast<std::size_t>(N) + static_cast<std::size_t>(k)]
                = coeffs[static_cast<std::size_t>(k)] * inv_denom;
        }

        // First derivative coefficients: d/dx (sum_k c_ik * x^k) = sum_{k>=1} k*c_ik * x^(k-1).
        if (N >= 2) {
            for (int k = 1; k < N; ++k) {
                axis_d_coeffs_[static_cast<std::size_t>(i) * static_cast<std::size_t>(N - 1)
                              + static_cast<std::size_t>(k - 1)]
                    = static_cast<Real>(k)
                      * axis_v_coeffs_[static_cast<std::size_t>(i) * static_cast<std::size_t>(N)
                                       + static_cast<std::size_t>(k)];
            }
        }

        // Second derivative coefficients: d^2/dx^2 = sum_{k>=2} k*(k-1)*c_ik * x^(k-2).
        if (N >= 3) {
            for (int k = 2; k < N; ++k) {
                axis_d2_coeffs_[static_cast<std::size_t>(i) * static_cast<std::size_t>(N - 2)
                              + static_cast<std::size_t>(k - 2)]
                    = static_cast<Real>(k * (k - 1))
                      * axis_v_coeffs_[static_cast<std::size_t>(i) * static_cast<std::size_t>(N)
                                       + static_cast<std::size_t>(k)];
            }
        }
    }
}

void LagrangeBasis::build_point_nodes() {
    nodes_.push_back(math::Vector<Real, 3>{Real(0), Real(0), Real(0)});
}

void LagrangeBasis::init_equispaced_1d_nodes() {
    nodes_1d_.clear();
    for (int i = 0; i <= std::max(order_, 0); ++i) {
        nodes_1d_.push_back(detail::equispaced_pm_one_coord(i, order_));
    }
}

void LagrangeBasis::build_tensor_product_nodes(int dimensions) {
    init_equispaced_1d_nodes();

    if (dimensions < 1 || dimensions > 3) {
        throw BasisConfigurationException("LagrangeBasis::build_tensor_product_nodes requires dimension 1, 2, or 3",
                                          __FILE__, __LINE__, __func__);
    }

    nodes_ = NodeOrdering::get_lagrange_node_coords(element_type_, order_);
    tensor_indices_.resize(nodes_.size(), TensorNodeIndex{0u, 0u, 0u});
    for (std::size_t n = 0; n < nodes_.size(); ++n) {
        tensor_indices_[n][0] = lattice_index_pm_one(
            nodes_[n][0], order_,
            "LagrangeBasis: invalid tensor-product x-coordinate in public node ordering");
        if (dimensions >= 2) {
            tensor_indices_[n][1] = lattice_index_pm_one(
                nodes_[n][1], order_,
                "LagrangeBasis: invalid tensor-product y-coordinate in public node ordering");
        }
        if (dimensions == 3) {
            tensor_indices_[n][2] = lattice_index_pm_one(
                nodes_[n][2], order_,
                "LagrangeBasis: invalid tensor-product z-coordinate in public node ordering");
        }
    }
}

void LagrangeBasis::build_simplex_nodes() {
    nodes_ = NodeOrdering::get_lagrange_node_coords(element_type_, order_);
    const LagrangeTopology topology = lagrange_topology_traits(element_type_).topology;
    simplex_exponents_.clear();
    simplex_exponents_.reserve(nodes_.size());
    for (const auto& node : nodes_) {
        switch (topology) {
            case LagrangeTopology::Triangle:
                simplex_exponents_.push_back(triangle_exponents_from_public_node(node, order_));
                break;
            case LagrangeTopology::Tetrahedron:
                simplex_exponents_.push_back(tetrahedron_exponents_from_public_node(node, order_));
                break;
            default:
                throw BasisElementCompatibilityException("LagrangeBasis::build_simplex_nodes requires simplex topology",
                                                         __FILE__, __LINE__, __func__);
        }
    }
}

void LagrangeBasis::build_wedge_nodes() {
    init_equispaced_1d_nodes();
    const auto triangle_nodes = NodeOrdering::get_lagrange_node_coords(ElementType::Triangle3, order_);
    simplex_exponents_.clear();
    simplex_exponents_.reserve(triangle_nodes.size());
    std::map<std::array<int, 4>, std::size_t> triangle_descriptor_to_index;
    for (std::size_t tri = 0; tri < triangle_nodes.size(); ++tri) {
        const auto exponents = triangle_exponents_from_public_node(triangle_nodes[tri], order_);
        simplex_exponents_.push_back(exponents);
        triangle_descriptor_to_index.emplace(exponents, tri);
    }

    nodes_ = NodeOrdering::get_lagrange_node_coords(element_type_, order_);
    wedge_indices_.clear();
    wedge_indices_.reserve(nodes_.size());
    for (const auto& node : nodes_) {
        const auto exponents = triangle_exponents_from_public_node(node, order_);
        const auto found = triangle_descriptor_to_index.find(exponents);
        if (found == triangle_descriptor_to_index.end()) {
            throw BasisNodeOrderingException("LagrangeBasis: failed to resolve wedge triangle descriptor in public ordering",
                                             __FILE__, __LINE__, __func__);
        }
        wedge_indices_.push_back(WedgeNodeIndex{
            found->second,
            lattice_index_pm_one(node[2], order_,
                                 "LagrangeBasis: invalid wedge z-coordinate in public node ordering")
        });
    }
}

void LagrangeBasis::build_pyramid_nodes() {
    nodes_ = detail::PyramidLagrangeCache::get(order_).nodes;
}

void LagrangeBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                    std::vector<Real>& values) const {
    values.resize(size());
    const LagrangeTopology topology = lagrange_topology_traits(element_type_).topology;
    const int n_axis = static_cast<int>(nodes_1d_.size());
    const Real* vc = axis_v_coeffs_.data();
    const Real* dc = axis_d_coeffs_.data();
    const Real* d2c = axis_d2_coeffs_.data();
    switch (topology) {
        case LagrangeTopology::Point:
            values[0] = Real(1);
            return;
        case LagrangeTopology::Line:
        case LagrangeTopology::Quadrilateral:
        case LagrangeTopology::Hexahedron: {
            // E4: hardcoded order-1 tensor-product fast paths (LINE2/QUAD4/HEX8).
            if (order_ == 1) {
                const Real lx = (Real(1) - xi[0]) * Real(0.5);
                const Real ux = (Real(1) + xi[0]) * Real(0.5);
                if (topology == LagrangeTopology::Line) {
                    values[0] = lx;
                    values[1] = ux;
                    return;
                }
                const Real ly = (Real(1) - xi[1]) * Real(0.5);
                const Real uy = (Real(1) + xi[1]) * Real(0.5);
                if (topology == LagrangeTopology::Quadrilateral) {
                    values[0] = lx * ly;
                    values[1] = ux * ly;
                    values[2] = ux * uy;
                    values[3] = lx * uy;
                    return;
                }
                const Real lz = (Real(1) - xi[2]) * Real(0.5);
                const Real uz = (Real(1) + xi[2]) * Real(0.5);
                const Real lxly = lx * ly;
                const Real uxly = ux * ly;
                const Real uxuy = ux * uy;
                const Real lxuy = lx * uy;
                values[0] = lxly * lz;
                values[1] = uxly * lz;
                values[2] = uxuy * lz;
                values[3] = lxuy * lz;
                values[4] = lxly * uz;
                values[5] = uxly * uz;
                values[6] = uxuy * uz;
                values[7] = lxuy * uz;
                return;
            }
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations x_axis =
                fill_axis_scratch(scratch.axis_x, vc, dc, d2c, n_axis, xi[0], AxisDeriv::ValuesOnly);
            AxisBasisEvaluations y_axis = constant_axis_basis();
            AxisBasisEvaluations z_axis = constant_axis_basis();

            if (topology != LagrangeTopology::Line) {
                y_axis = fill_axis_scratch(scratch.axis_y, vc, dc, d2c, n_axis, xi[1], AxisDeriv::ValuesOnly);
            }
            if (topology == LagrangeTopology::Hexahedron) {
                z_axis = fill_axis_scratch(scratch.axis_z, vc, dc, d2c, n_axis, xi[2], AxisDeriv::ValuesOnly);
            }

            evaluate_tensor_product_values(tensor_indices_, x_axis, y_axis, z_axis, values);
            return;
        }
        case LagrangeTopology::Wedge: {
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations z_axis =
                fill_axis_scratch(scratch.axis_z, vc, dc, d2c, n_axis, xi[2], AxisDeriv::ValuesOnly);
            detail::evaluate_triangle_simplex_basis(simplex_exponents_, order_, xi,
                                                    &scratch.tri_values, nullptr, nullptr);

            for (std::size_t n = 0; n < wedge_indices_.size(); ++n) {
                const auto& index = wedge_indices_[n];
                values[n] = scratch.tri_values[index[0]] * z_axis.values[index[1]];
            }
            return;
        }
        case LagrangeTopology::Pyramid: {
            const auto& data = detail::PyramidLagrangeCache::get(order_);
            detail::PyramidLagrangeCache::evaluate_values(data, xi, values);
            return;
        }
        case LagrangeTopology::Triangle:
            // D1: TRI3 hardcoded fast path (order 1).
            if (order_ == 1) {
                values[0] = Real(1) - xi[0] - xi[1];
                values[1] = xi[0];
                values[2] = xi[1];
                return;
            }
            // E1: TRI6 hardcoded fast path (order 2).
            if (order_ == 2) {
                evaluate_tri6_p2_fast(xi, simplex_exponents_, &values, nullptr, nullptr);
                return;
            }
            detail::evaluate_triangle_simplex_basis(simplex_exponents_, order_, xi, &values, nullptr, nullptr);
            return;
        case LagrangeTopology::Tetrahedron:
            // D1: TET4 hardcoded fast path (order 1).
            if (order_ == 1) {
                values[0] = Real(1) - xi[0] - xi[1] - xi[2];
                values[1] = xi[0];
                values[2] = xi[1];
                values[3] = xi[2];
                return;
            }
            // E1: TET10 hardcoded fast path (order 2).
            if (order_ == 2) {
                evaluate_tet10_p2_fast(xi, simplex_exponents_, &values, nullptr, nullptr);
                return;
            }
            detail::evaluate_tetrahedron_simplex_basis(simplex_exponents_, order_, xi, &values, nullptr, nullptr);
            return;
    }

    throw BasisEvaluationException("Unsupported element in evaluate_values",
                                   __FILE__, __LINE__, __func__);
}

void LagrangeBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                       std::vector<Gradient>& gradients) const {
    const LagrangeTopology topology = lagrange_topology_traits(element_type_).topology;
    const int n_axis = static_cast<int>(nodes_1d_.size());
    const Real* vc = axis_v_coeffs_.data();
    const Real* dc = axis_d_coeffs_.data();
    const Real* d2c = axis_d2_coeffs_.data();
    switch (topology) {
        case LagrangeTopology::Point:
            gradients.resize(size());
            gradients[0] = Gradient{};
            return;
        case LagrangeTopology::Line:
        case LagrangeTopology::Quadrilateral:
        case LagrangeTopology::Hexahedron: {
            // E4: hardcoded order-1 tensor-product fast paths.
            if (order_ == 1) {
                const Real lx = (Real(1) - xi[0]) * Real(0.5);
                const Real ux = (Real(1) + xi[0]) * Real(0.5);
                if (topology == LagrangeTopology::Line) {
                    gradients.resize(2);
                    gradients[0] = Gradient{Real(-0.5), Real(0), Real(0)};
                    gradients[1] = Gradient{Real( 0.5), Real(0), Real(0)};
                    return;
                }
                const Real ly = (Real(1) - xi[1]) * Real(0.5);
                const Real uy = (Real(1) + xi[1]) * Real(0.5);
                if (topology == LagrangeTopology::Quadrilateral) {
                    gradients.resize(4);
                    gradients[0] = Gradient{Real(-0.5)*ly, Real(-0.5)*lx, Real(0)};
                    gradients[1] = Gradient{Real( 0.5)*ly, Real(-0.5)*ux, Real(0)};
                    gradients[2] = Gradient{Real( 0.5)*uy, Real( 0.5)*ux, Real(0)};
                    gradients[3] = Gradient{Real(-0.5)*uy, Real( 0.5)*lx, Real(0)};
                    return;
                }
                const Real lz = (Real(1) - xi[2]) * Real(0.5);
                const Real uz = (Real(1) + xi[2]) * Real(0.5);
                gradients.resize(8);
                gradients[0] = Gradient{Real(-0.5)*ly*lz, Real(-0.5)*lx*lz, Real(-0.5)*lx*ly};
                gradients[1] = Gradient{Real( 0.5)*ly*lz, Real(-0.5)*ux*lz, Real(-0.5)*ux*ly};
                gradients[2] = Gradient{Real( 0.5)*uy*lz, Real( 0.5)*ux*lz, Real(-0.5)*ux*uy};
                gradients[3] = Gradient{Real(-0.5)*uy*lz, Real( 0.5)*lx*lz, Real(-0.5)*lx*uy};
                gradients[4] = Gradient{Real(-0.5)*ly*uz, Real(-0.5)*lx*uz, Real( 0.5)*lx*ly};
                gradients[5] = Gradient{Real( 0.5)*ly*uz, Real(-0.5)*ux*uz, Real( 0.5)*ux*ly};
                gradients[6] = Gradient{Real( 0.5)*uy*uz, Real( 0.5)*ux*uz, Real( 0.5)*ux*uy};
                gradients[7] = Gradient{Real(-0.5)*uy*uz, Real( 0.5)*lx*uz, Real( 0.5)*lx*uy};
                return;
            }
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations x_axis =
                fill_axis_scratch(scratch.axis_x, vc, dc, d2c, n_axis, xi[0], AxisDeriv::ValuesAndFirst);
            AxisBasisEvaluations y_axis = constant_axis_basis();
            AxisBasisEvaluations z_axis = constant_axis_basis();

            if (topology != LagrangeTopology::Line) {
                y_axis = fill_axis_scratch(scratch.axis_y, vc, dc, d2c, n_axis, xi[1], AxisDeriv::ValuesAndFirst);
            }
            if (topology == LagrangeTopology::Hexahedron) {
                z_axis = fill_axis_scratch(scratch.axis_z, vc, dc, d2c, n_axis, xi[2], AxisDeriv::ValuesAndFirst);
            }

            evaluate_tensor_product_gradients(tensor_indices_, x_axis, y_axis, z_axis, gradients);
            return;
        }
        case LagrangeTopology::Wedge: {
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations z_axis =
                fill_axis_scratch(scratch.axis_z, vc, dc, d2c, n_axis, xi[2], AxisDeriv::ValuesAndFirst);
            detail::evaluate_triangle_simplex_basis(
                simplex_exponents_, order_, xi,
                &scratch.tri_values, &scratch.tri_gradients, nullptr);

            gradients.resize(wedge_indices_.size());
            for (std::size_t n = 0; n < wedge_indices_.size(); ++n) {
                const auto& index = wedge_indices_[n];
                gradients[n][0] = scratch.tri_gradients[index[0]][0] * z_axis.values[index[1]];
                gradients[n][1] = scratch.tri_gradients[index[0]][1] * z_axis.values[index[1]];
                gradients[n][2] = scratch.tri_values[index[0]] * z_axis.first[index[1]];
            }
            return;
        }
        case LagrangeTopology::Pyramid: {
            const auto& data = detail::PyramidLagrangeCache::get(order_);
            detail::PyramidLagrangeCache::evaluate_gradients(data, xi, gradients);
            return;
        }
        case LagrangeTopology::Triangle:
            // D1: TRI3 hardcoded fast path (order 1).
            if (order_ == 1) {
                gradients.resize(3);
                gradients[0] = Gradient{Real(-1), Real(-1), Real(0)};
                gradients[1] = Gradient{Real(1),  Real(0),  Real(0)};
                gradients[2] = Gradient{Real(0),  Real(1),  Real(0)};
                return;
            }
            // E1: TRI6 hardcoded fast path (order 2).
            if (order_ == 2) {
                evaluate_tri6_p2_fast(xi, simplex_exponents_, nullptr, &gradients, nullptr);
                return;
            }
            detail::evaluate_triangle_simplex_basis(simplex_exponents_, order_, xi, nullptr, &gradients, nullptr);
            return;
        case LagrangeTopology::Tetrahedron:
            // D1: TET4 hardcoded fast path (order 1).
            if (order_ == 1) {
                gradients.resize(4);
                gradients[0] = Gradient{Real(-1), Real(-1), Real(-1)};
                gradients[1] = Gradient{Real(1),  Real(0),  Real(0)};
                gradients[2] = Gradient{Real(0),  Real(1),  Real(0)};
                gradients[3] = Gradient{Real(0),  Real(0),  Real(1)};
                return;
            }
            // E1: TET10 hardcoded fast path (order 2).
            if (order_ == 2) {
                evaluate_tet10_p2_fast(xi, simplex_exponents_, nullptr, &gradients, nullptr);
                return;
            }
            detail::evaluate_tetrahedron_simplex_basis(simplex_exponents_, order_, xi, nullptr, &gradients, nullptr);
            return;
    }

    throw BasisEvaluationException("Unsupported element in evaluate_gradients",
                                   __FILE__, __LINE__, __func__);
}

void LagrangeBasis::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                      std::vector<Hessian>& hessians) const {
    const LagrangeTopology topology = lagrange_topology_traits(element_type_).topology;
    const int n_axis = static_cast<int>(nodes_1d_.size());
    const Real* vc = axis_v_coeffs_.data();
    const Real* dc = axis_d_coeffs_.data();
    const Real* d2c = axis_d2_coeffs_.data();
    switch (topology) {
        case LagrangeTopology::Point:
            hessians.resize(size());
            hessians[0] = Hessian{};
            return;
        case LagrangeTopology::Line:
        case LagrangeTopology::Quadrilateral:
        case LagrangeTopology::Hexahedron: {
            // E4: hardcoded order-1 tensor-product fast paths.
            if (order_ == 1) {
                if (topology == LagrangeTopology::Line) {
                    hessians.assign(2, Hessian{});
                    return;
                }
                const Real lx = (Real(1) - xi[0]) * Real(0.5);
                const Real ux = (Real(1) + xi[0]) * Real(0.5);
                const Real ly = (Real(1) - xi[1]) * Real(0.5);
                const Real uy = (Real(1) + xi[1]) * Real(0.5);
                if (topology == LagrangeTopology::Quadrilateral) {
                    hessians.assign(4, Hessian{});
                    constexpr Real qrt = Real(0.25);
                    // d²N/dxdy = sign_x * sign_y * 0.25.
                    hessians[0](0,1) =  qrt; hessians[0](1,0) =  qrt;  // (-,-)
                    hessians[1](0,1) = -qrt; hessians[1](1,0) = -qrt;  // (+,-)
                    hessians[2](0,1) =  qrt; hessians[2](1,0) =  qrt;  // (+,+)
                    hessians[3](0,1) = -qrt; hessians[3](1,0) = -qrt;  // (-,+)
                    return;
                }
                // Hex8 — diagonal entries are zero (each axis basis is linear).
                const Real lz = (Real(1) - xi[2]) * Real(0.5);
                const Real uz = (Real(1) + xi[2]) * Real(0.5);
                hessians.assign(8, Hessian{});
                constexpr Real qrt = Real(0.25);
                // VTK Hex8 sign tuples per node:
                //   0:(-,-,-), 1:(+,-,-), 2:(+,+,-), 3:(-,+,-),
                //   4:(-,-,+), 5:(+,-,+), 6:(+,+,+), 7:(-,+,+).
                // Per-node L_a(x), L_b(y), L_c(z) values:
                const Real ax[8] = {lx, ux, ux, lx, lx, ux, ux, lx};
                const Real ay[8] = {ly, ly, uy, uy, ly, ly, uy, uy};
                const Real az[8] = {lz, lz, lz, lz, uz, uz, uz, uz};
                const int sx[8] = {-1, 1, 1, -1, -1, 1, 1, -1};
                const int sy[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
                const int sz[8] = {-1, -1, -1, -1, 1, 1, 1, 1};
                for (int n = 0; n < 8; ++n) {
                    Hessian& H = hessians[n];
                    H(0, 1) = static_cast<Real>(sx[n] * sy[n]) * qrt * az[n];
                    H(1, 0) = H(0, 1);
                    H(0, 2) = static_cast<Real>(sx[n] * sz[n]) * qrt * ay[n];
                    H(2, 0) = H(0, 2);
                    H(1, 2) = static_cast<Real>(sy[n] * sz[n]) * qrt * ax[n];
                    H(2, 1) = H(1, 2);
                }
                return;
            }
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations x_axis =
                fill_axis_scratch(scratch.axis_x, vc, dc, d2c, n_axis, xi[0], AxisDeriv::ValuesAndFirstAndSecond);
            AxisBasisEvaluations y_axis = constant_axis_basis();
            AxisBasisEvaluations z_axis = constant_axis_basis();

            if (topology != LagrangeTopology::Line) {
                y_axis = fill_axis_scratch(scratch.axis_y, vc, dc, d2c, n_axis, xi[1], AxisDeriv::ValuesAndFirstAndSecond);
            }
            if (topology == LagrangeTopology::Hexahedron) {
                z_axis = fill_axis_scratch(scratch.axis_z, vc, dc, d2c, n_axis, xi[2], AxisDeriv::ValuesAndFirstAndSecond);
            }

            evaluate_tensor_product_hessians(tensor_indices_, x_axis, y_axis, z_axis, hessians);
            return;
        }
        case LagrangeTopology::Triangle:
            // D1: TRI3 hardcoded fast path (order 1).
            if (order_ == 1) {
                hessians.assign(3, Hessian{});
                return;
            }
            // E1: TRI6 hardcoded fast path (order 2).
            if (order_ == 2) {
                evaluate_tri6_p2_fast(xi, simplex_exponents_, nullptr, nullptr, &hessians);
                return;
            }
            detail::evaluate_triangle_simplex_basis(simplex_exponents_, order_, xi, nullptr, nullptr, &hessians);
            return;
        case LagrangeTopology::Tetrahedron:
            // D1: TET4 hardcoded fast path (order 1).
            if (order_ == 1) {
                hessians.assign(4, Hessian{});
                return;
            }
            // E1: TET10 hardcoded fast path (order 2).
            if (order_ == 2) {
                evaluate_tet10_p2_fast(xi, simplex_exponents_, nullptr, nullptr, &hessians);
                return;
            }
            detail::evaluate_tetrahedron_simplex_basis(simplex_exponents_, order_, xi, nullptr, nullptr, &hessians);
            return;
        case LagrangeTopology::Wedge: {
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations z_axis =
                fill_axis_scratch(scratch.axis_z, vc, dc, d2c, n_axis, xi[2], AxisDeriv::ValuesAndFirstAndSecond);
            detail::evaluate_triangle_simplex_basis(
                simplex_exponents_, order_, xi,
                &scratch.tri_values, &scratch.tri_gradients, &scratch.tri_hessians);

            hessians.resize(wedge_indices_.size());
            for (std::size_t n = 0; n < wedge_indices_.size(); ++n) {
                const auto& index = wedge_indices_[n];
                Hessian H{};
                H(0, 0) = scratch.tri_hessians[index[0]](0, 0) * z_axis.values[index[1]];
                H(1, 1) = scratch.tri_hessians[index[0]](1, 1) * z_axis.values[index[1]];
                H(0, 1) = scratch.tri_hessians[index[0]](0, 1) * z_axis.values[index[1]];
                H(1, 0) = H(0, 1);

                H(2, 2) = scratch.tri_values[index[0]] * z_axis.second[index[1]];

                H(0, 2) = scratch.tri_gradients[index[0]][0] * z_axis.first[index[1]];
                H(2, 0) = H(0, 2);
                H(1, 2) = scratch.tri_gradients[index[0]][1] * z_axis.first[index[1]];
                H(2, 1) = H(1, 2);

                hessians[n] = H;
            }
            return;
        }
        case LagrangeTopology::Pyramid: {
            const auto& data = detail::PyramidLagrangeCache::get(order_);
            detail::PyramidLagrangeCache::evaluate_hessians(data, xi, hessians);
            return;
        }
    }

    throw BasisEvaluationException("Unsupported element in evaluate_hessians",
                                   __FILE__, __LINE__, __func__);
}

void LagrangeBasis::evaluate_all(const math::Vector<Real, 3>& xi,
                                 std::vector<Real>& values,
                                 std::vector<Gradient>& gradients,
                                 std::vector<Hessian>& hessians) const {
    const LagrangeTopology topology = lagrange_topology_traits(element_type_).topology;
    const int n_axis = static_cast<int>(nodes_1d_.size());
    const Real* vc = axis_v_coeffs_.data();
    const Real* dc = axis_d_coeffs_.data();
    const Real* d2c = axis_d2_coeffs_.data();
    switch (topology) {
        case LagrangeTopology::Point:
            values.resize(size());
            values[0] = Real(1);
            gradients.resize(size());
            gradients[0] = Gradient{};
            hessians.resize(size());
            hessians[0] = Hessian{};
            return;
        case LagrangeTopology::Line:
        case LagrangeTopology::Quadrilateral:
        case LagrangeTopology::Hexahedron: {
            // E4: hardcoded order-1 tensor-product fast paths (fused values+grad+hess).
            if (order_ == 1) {
                const Real lx = (Real(1) - xi[0]) * Real(0.5);
                const Real ux = (Real(1) + xi[0]) * Real(0.5);
                if (topology == LagrangeTopology::Line) {
                    values.resize(2);
                    values[0] = lx; values[1] = ux;
                    gradients.resize(2);
                    gradients[0] = Gradient{Real(-0.5), Real(0), Real(0)};
                    gradients[1] = Gradient{Real( 0.5), Real(0), Real(0)};
                    hessians.assign(2, Hessian{});
                    return;
                }
                const Real ly = (Real(1) - xi[1]) * Real(0.5);
                const Real uy = (Real(1) + xi[1]) * Real(0.5);
                if (topology == LagrangeTopology::Quadrilateral) {
                    values.resize(4);
                    values[0] = lx*ly; values[1] = ux*ly; values[2] = ux*uy; values[3] = lx*uy;
                    gradients.resize(4);
                    gradients[0] = Gradient{Real(-0.5)*ly, Real(-0.5)*lx, Real(0)};
                    gradients[1] = Gradient{Real( 0.5)*ly, Real(-0.5)*ux, Real(0)};
                    gradients[2] = Gradient{Real( 0.5)*uy, Real( 0.5)*ux, Real(0)};
                    gradients[3] = Gradient{Real(-0.5)*uy, Real( 0.5)*lx, Real(0)};
                    hessians.assign(4, Hessian{});
                    constexpr Real qrt = Real(0.25);
                    hessians[0](0,1) =  qrt; hessians[0](1,0) =  qrt;
                    hessians[1](0,1) = -qrt; hessians[1](1,0) = -qrt;
                    hessians[2](0,1) =  qrt; hessians[2](1,0) =  qrt;
                    hessians[3](0,1) = -qrt; hessians[3](1,0) = -qrt;
                    return;
                }
                // Hex8.
                const Real lz = (Real(1) - xi[2]) * Real(0.5);
                const Real uz = (Real(1) + xi[2]) * Real(0.5);
                values.resize(8);
                values[0] = lx*ly*lz; values[1] = ux*ly*lz; values[2] = ux*uy*lz; values[3] = lx*uy*lz;
                values[4] = lx*ly*uz; values[5] = ux*ly*uz; values[6] = ux*uy*uz; values[7] = lx*uy*uz;
                gradients.resize(8);
                gradients[0] = Gradient{Real(-0.5)*ly*lz, Real(-0.5)*lx*lz, Real(-0.5)*lx*ly};
                gradients[1] = Gradient{Real( 0.5)*ly*lz, Real(-0.5)*ux*lz, Real(-0.5)*ux*ly};
                gradients[2] = Gradient{Real( 0.5)*uy*lz, Real( 0.5)*ux*lz, Real(-0.5)*ux*uy};
                gradients[3] = Gradient{Real(-0.5)*uy*lz, Real( 0.5)*lx*lz, Real(-0.5)*lx*uy};
                gradients[4] = Gradient{Real(-0.5)*ly*uz, Real(-0.5)*lx*uz, Real( 0.5)*lx*ly};
                gradients[5] = Gradient{Real( 0.5)*ly*uz, Real(-0.5)*ux*uz, Real( 0.5)*ux*ly};
                gradients[6] = Gradient{Real( 0.5)*uy*uz, Real( 0.5)*ux*uz, Real( 0.5)*ux*uy};
                gradients[7] = Gradient{Real(-0.5)*uy*uz, Real( 0.5)*lx*uz, Real( 0.5)*lx*uy};
                hessians.assign(8, Hessian{});
                constexpr Real qrt = Real(0.25);
                const Real ax[8] = {lx, ux, ux, lx, lx, ux, ux, lx};
                const Real ay[8] = {ly, ly, uy, uy, ly, ly, uy, uy};
                const Real az[8] = {lz, lz, lz, lz, uz, uz, uz, uz};
                const int sx[8] = {-1, 1, 1, -1, -1, 1, 1, -1};
                const int sy[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
                const int sz[8] = {-1, -1, -1, -1, 1, 1, 1, 1};
                for (int n = 0; n < 8; ++n) {
                    Hessian& H = hessians[n];
                    H(0, 1) = static_cast<Real>(sx[n] * sy[n]) * qrt * az[n]; H(1, 0) = H(0, 1);
                    H(0, 2) = static_cast<Real>(sx[n] * sz[n]) * qrt * ay[n]; H(2, 0) = H(0, 2);
                    H(1, 2) = static_cast<Real>(sy[n] * sz[n]) * qrt * ax[n]; H(2, 1) = H(1, 2);
                }
                return;
            }
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations x_axis =
                fill_axis_scratch(scratch.axis_x, vc, dc, d2c, n_axis, xi[0], AxisDeriv::ValuesAndFirstAndSecond);
            AxisBasisEvaluations y_axis = constant_axis_basis();
            AxisBasisEvaluations z_axis = constant_axis_basis();

            if (topology != LagrangeTopology::Line) {
                y_axis = fill_axis_scratch(scratch.axis_y, vc, dc, d2c, n_axis, xi[1], AxisDeriv::ValuesAndFirstAndSecond);
            }
            if (topology == LagrangeTopology::Hexahedron) {
                z_axis = fill_axis_scratch(scratch.axis_z, vc, dc, d2c, n_axis, xi[2], AxisDeriv::ValuesAndFirstAndSecond);
            }

            evaluate_tensor_product_factorized(tensor_indices_, x_axis, y_axis, z_axis,
                                               &values, &gradients, &hessians);
            return;
        }
        case LagrangeTopology::Triangle:
            // D1: TRI3 hardcoded fast path (order 1).
            if (order_ == 1) {
                values.resize(3);
                values[0] = Real(1) - xi[0] - xi[1];
                values[1] = xi[0];
                values[2] = xi[1];
                gradients.resize(3);
                gradients[0] = Gradient{Real(-1), Real(-1), Real(0)};
                gradients[1] = Gradient{Real(1),  Real(0),  Real(0)};
                gradients[2] = Gradient{Real(0),  Real(1),  Real(0)};
                hessians.assign(3, Hessian{});
                return;
            }
            // E1: TRI6 hardcoded fast path (order 2).
            if (order_ == 2) {
                evaluate_tri6_p2_fast(xi, simplex_exponents_, &values, &gradients, &hessians);
                return;
            }
            detail::evaluate_triangle_simplex_basis(simplex_exponents_, order_, xi,
                                                    &values, &gradients, &hessians);
            return;
        case LagrangeTopology::Tetrahedron:
            // D1: TET4 hardcoded fast path (order 1).
            if (order_ == 1) {
                values.resize(4);
                values[0] = Real(1) - xi[0] - xi[1] - xi[2];
                values[1] = xi[0];
                values[2] = xi[1];
                values[3] = xi[2];
                gradients.resize(4);
                gradients[0] = Gradient{Real(-1), Real(-1), Real(-1)};
                gradients[1] = Gradient{Real(1),  Real(0),  Real(0)};
                gradients[2] = Gradient{Real(0),  Real(1),  Real(0)};
                gradients[3] = Gradient{Real(0),  Real(0),  Real(1)};
                hessians.assign(4, Hessian{});
                return;
            }
            // E1: TET10 hardcoded fast path (order 2).
            if (order_ == 2) {
                evaluate_tet10_p2_fast(xi, simplex_exponents_, &values, &gradients, &hessians);
                return;
            }
            detail::evaluate_tetrahedron_simplex_basis(simplex_exponents_, order_, xi,
                                                       &values, &gradients, &hessians);
            return;
        case LagrangeTopology::Wedge: {
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations z_axis =
                fill_axis_scratch(scratch.axis_z, vc, dc, d2c, n_axis, xi[2], AxisDeriv::ValuesAndFirstAndSecond);
            detail::evaluate_triangle_simplex_basis(
                simplex_exponents_, order_, xi,
                &scratch.tri_values, &scratch.tri_gradients, &scratch.tri_hessians);

            const std::size_t n_nodes = wedge_indices_.size();
            values.resize(n_nodes);
            gradients.resize(n_nodes);
            hessians.resize(n_nodes);

            for (std::size_t n = 0; n < n_nodes; ++n) {
                const auto& index = wedge_indices_[n];
                const std::size_t tri_idx = index[0];
                const std::size_t z_idx = index[1];
                const Real zv  = z_axis.values[z_idx];
                const Real zd  = z_axis.first[z_idx];
                const Real zd2 = z_axis.second[z_idx];

                values[n] = scratch.tri_values[tri_idx] * zv;

                gradients[n][0] = scratch.tri_gradients[tri_idx][0] * zv;
                gradients[n][1] = scratch.tri_gradients[tri_idx][1] * zv;
                gradients[n][2] = scratch.tri_values[tri_idx] * zd;

                Hessian H{};
                H(0, 0) = scratch.tri_hessians[tri_idx](0, 0) * zv;
                H(1, 1) = scratch.tri_hessians[tri_idx](1, 1) * zv;
                H(0, 1) = scratch.tri_hessians[tri_idx](0, 1) * zv;
                H(1, 0) = H(0, 1);

                H(2, 2) = scratch.tri_values[tri_idx] * zd2;

                H(0, 2) = scratch.tri_gradients[tri_idx][0] * zd;
                H(2, 0) = H(0, 2);
                H(1, 2) = scratch.tri_gradients[tri_idx][1] * zd;
                H(2, 1) = H(1, 2);

                hessians[n] = H;
            }
            return;
        }
        case LagrangeTopology::Pyramid: {
            const auto& data = detail::PyramidLagrangeCache::get(order_);
            detail::PyramidLagrangeCache::evaluate_values(data, xi, values);
            detail::PyramidLagrangeCache::evaluate_gradients(data, xi, gradients);
            detail::PyramidLagrangeCache::evaluate_hessians(data, xi, hessians);
            return;
        }
    }

    throw BasisEvaluationException("Unsupported element in evaluate_all",
                                   __FILE__, __LINE__, __func__);
}

// D3: raw-buffer overrides. Reuse the per-thread scratch vectors so the user's
// pre-sized buffer doesn't need to grow, and copy out at the end.
void LagrangeBasis::evaluate_values_to(const math::Vector<Real, 3>& xi,
                                       Real* values_out) const {
    static thread_local std::vector<Real> tmp;
    tmp.resize(size());
    evaluate_values(xi, tmp);
    for (std::size_t i = 0; i < tmp.size(); ++i) values_out[i] = tmp[i];
}

void LagrangeBasis::evaluate_gradients_to(const math::Vector<Real, 3>& xi,
                                          Real* gradients_out) const {
    static thread_local std::vector<Gradient> tmp;
    tmp.resize(size());
    evaluate_gradients(xi, tmp);
    for (std::size_t i = 0; i < tmp.size(); ++i) {
        gradients_out[i * 3 + 0] = tmp[i][0];
        gradients_out[i * 3 + 1] = tmp[i][1];
        gradients_out[i * 3 + 2] = tmp[i][2];
    }
}

void LagrangeBasis::evaluate_hessians_to(const math::Vector<Real, 3>& xi,
                                         Real* hessians_out) const {
    static thread_local std::vector<Hessian> tmp;
    tmp.resize(size());
    evaluate_hessians(xi, tmp);
    for (std::size_t i = 0; i < tmp.size(); ++i) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                hessians_out[i * 9 + static_cast<std::size_t>(r * 3 + c)] =
                    tmp[i](static_cast<std::size_t>(r), static_cast<std::size_t>(c));
            }
        }
    }
}

// C3+C4: Multi-QP entry method that writes directly to SoA output buffers.
// Caller passes flat DOF-major buffers; we evaluate per QP and scatter-write
// without going through a vector<Gradient>/<Hessian> nested-layout transpose.
void LagrangeBasis::evaluate_at_quadrature_points(
    const std::vector<math::Vector<Real, 3>>& points,
    Real* values_out,
    Real* gradients_out,
    Real* hessians_out) const {
    const std::size_t num_qpts = points.size();
    const std::size_t num_dofs = size();

    // Per-QP scratch buffers (reused across QPs).
    static thread_local std::vector<Real> v_tmp;
    static thread_local std::vector<Gradient> g_tmp;
    static thread_local std::vector<Hessian> h_tmp;

    if (values_out)    v_tmp.resize(num_dofs);
    if (gradients_out) g_tmp.resize(num_dofs);
    if (hessians_out)  h_tmp.resize(num_dofs);

    for (std::size_t q = 0; q < num_qpts; ++q) {
        if (values_out && gradients_out && hessians_out) {
            evaluate_all(points[q], v_tmp, g_tmp, h_tmp);
        } else {
            if (values_out)    evaluate_values(points[q], v_tmp);
            if (gradients_out) evaluate_gradients(points[q], g_tmp);
            if (hessians_out)  evaluate_hessians(points[q], h_tmp);
        }

        if (values_out) {
            for (std::size_t d = 0; d < num_dofs; ++d) {
                values_out[d * num_qpts + q] = v_tmp[d];
            }
        }
        if (gradients_out) {
            for (std::size_t d = 0; d < num_dofs; ++d) {
                gradients_out[(d * 3 + 0) * num_qpts + q] = g_tmp[d][0];
                gradients_out[(d * 3 + 1) * num_qpts + q] = g_tmp[d][1];
                gradients_out[(d * 3 + 2) * num_qpts + q] = g_tmp[d][2];
            }
        }
        if (hessians_out) {
            for (std::size_t d = 0; d < num_dofs; ++d) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        hessians_out[(d * 9 + static_cast<std::size_t>(r * 3 + c)) * num_qpts + q] =
                            h_tmp[d](static_cast<std::size_t>(r), static_cast<std::size_t>(c));
                    }
                }
            }
        }
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
