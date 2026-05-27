#include "VectorBasisRtConstruction.h"
#include "Basis/BasisTraits.h"
#include "Basis/LagrangeBasis.h"
#include "Basis/NodeOrderingConventions.h"
#include "Math/DenseLinearAlgebra.h"
#include "Elements/ReferenceElement.h"
#include "Quadrature/QuadratureFactory.h"
#include "VectorBasisDirectSeeds.h"
#include "VectorBasisEvaluationHelpers.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#ifdef FE_CHECK_ARG
#undef FE_CHECK_ARG
#endif
#define FE_CHECK_ARG(condition, message) BASIS_CHECK_CONSTRUCTION((condition), (message))

namespace svmp {
namespace FE {
namespace basis {
namespace detail {
namespace vector_construction {

using vector_common::bilinear;
using vector_common::bilinear_du;
using vector_common::bilinear_dw;
using vector_common::cross3;
using vector_common::lerp;
using vector_common::normalize3;

std::vector<Real> invert_dense_matrix(std::vector<Real> A, std::size_t n) {
    return math::invert_dense_matrix(std::move(A), n, "VectorBasis moment matrix");
}

void fill_powers(Real x, int max_p, std::vector<Real>& out) {
    FE_CHECK_ARG(max_p >= 0, "powers: negative max_p");
    out.assign(static_cast<std::size_t>(max_p + 1), Real(1));
    for (int i = 1; i <= max_p; ++i) {
        out[static_cast<std::size_t>(i)] = out[static_cast<std::size_t>(i - 1)] * x;
    }
}

Real eval_transformed_nd_monomial_scalar(const std::array<int, 4>& mono,
                                         const std::vector<Real>& px,
                                         const std::vector<Real>& py,
                                         const std::vector<Real>& pz) {
    return px[static_cast<std::size_t>(mono[1])] *
           py[static_cast<std::size_t>(mono[2])] *
           pz[static_cast<std::size_t>(mono[3])];
}

// Compute rank of matrix A (n x n) via pivoted Gaussian elimination
int compute_rank(std::vector<Real> A, std::size_t n) {
    return static_cast<int>(math::dense_matrix_rank(std::move(A), n, n));
}

std::vector<Real> right_inverse_full_row_rank(const std::vector<Real>& A,
                                              std::size_t rows,
                                              std::size_t cols) {
    FE_CHECK_ARG(A.size() == rows * cols, "right_inverse_full_row_rank: size mismatch");
    const Real eps = math::dense_matrix_pivot_tolerance(
        rows, cols, math::dense_matrix_max_abs(A));
    std::vector<Real> work = A;
    std::vector<std::size_t> column_perm(cols);
    for (std::size_t c = 0; c < cols; ++c) {
        column_perm[c] = c;
    }

    for (std::size_t k = 0; k < rows; ++k) {
        std::size_t best_row = k;
        std::size_t best_col = k;
        Real best_abs = Real(0);
        for (std::size_t r = k; r < rows; ++r) {
            for (std::size_t c = k; c < cols; ++c) {
                const Real v = std::abs(work[r * cols + c]);
                if (v > best_abs) {
                    best_abs = v;
                    best_row = r;
                    best_col = c;
                }
            }
        }

        FE_CHECK_ARG(best_abs > eps,
                     "right_inverse_full_row_rank: candidate matrix is rank-deficient "
                     "(pivot " + std::to_string(k) + " below tolerance)");

        if (best_row != k) {
            for (std::size_t c = 0; c < cols; ++c) {
                std::swap(work[k * cols + c], work[best_row * cols + c]);
            }
        }
        if (best_col != k) {
            for (std::size_t r = 0; r < rows; ++r) {
                std::swap(work[r * cols + k], work[r * cols + best_col]);
            }
            std::swap(column_perm[k], column_perm[best_col]);
        }

        const Real pivot = work[k * cols + k];
        for (std::size_t r = k + 1; r < rows; ++r) {
            const Real factor = work[r * cols + k] / pivot;
            if (std::abs(factor) <= eps) {
                continue;
            }
            work[r * cols + k] = Real(0);
            for (std::size_t c = k + 1; c < cols; ++c) {
                work[r * cols + c] -= factor * work[k * cols + c];
            }
        }
    }

    std::vector<Real> square(rows * rows, Real(0));
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < rows; ++c) {
            square[r * rows + c] = A[r * cols + column_perm[c]];
        }
    }

    const int rank = compute_rank(square, rows);
    FE_CHECK_ARG(rank == static_cast<int>(rows),
                 "right_inverse_full_row_rank: selected square submatrix is rank-deficient "
                 "(rank " + std::to_string(rank) + " of " + std::to_string(rows) + ")");

    const std::vector<Real> square_inv = invert_dense_matrix(std::move(square), rows);
    std::vector<Real> result(cols * rows, Real(0));
    for (std::size_t c = 0; c < rows; ++c) {
        const std::size_t original_col = column_perm[c];
        for (std::size_t j = 0; j < rows; ++j) {
            result[original_col * rows + j] = square_inv[c * rows + j];
        }
    }
    return result;
}

std::vector<Real> rank_revealing_pseudo_inverse_dense_matrix(
    const std::vector<Real>& A,
    std::size_t n) {
    const auto result =
        math::rank_revealing_pseudo_inverse(A, n, n, "VectorBasis moment matrix");
    FE_CHECK_ARG(result.rank > 0,
                 "rank_revealing_pseudo_inverse_dense_matrix: moment matrix has numerical rank zero");
    return result.inverse;
}

void eval_rt_seed_values(ElementType type,
                         int order,
                         const Vec3& xi,
                         std::vector<Vec3>& values) {
    FE_CHECK_ARG(order == 1 || order == 2,
                 "eval_rt_seed_values: only wedge/pyramid RT(1-2) seeds are supported");
    if (is_wedge(type)) {
        if (order == 1) {
            vector_direct::eval_wedge_rt1_values(xi, values);
        } else {
            vector_direct::eval_wedge_rt2_values(xi, values);
        }
    } else {
        if (order == 1) {
            vector_direct::eval_pyramid_rt1_values(xi, values);
        } else {
            vector_direct::eval_pyramid_rt2_values(xi, values);
        }
    }
}

void eval_rt_seed_divergence(ElementType type,
                             int order,
                             const Vec3& xi,
                             std::vector<Real>& divergence) {
    FE_CHECK_ARG(order == 1 || order == 2,
                 "eval_rt_seed_divergence: only wedge/pyramid RT(1-2) seeds are supported");
    if (is_wedge(type)) {
        if (order == 1) {
            vector_direct::eval_wedge_rt1_divergence(xi, divergence);
        } else {
            vector_direct::eval_wedge_rt2_divergence(xi, divergence);
        }
    } else {
        if (order == 1) {
            vector_direct::eval_pyramid_rt1_divergence(xi, divergence);
        } else {
            vector_direct::eval_pyramid_rt2_divergence(xi, divergence);
        }
    }
}

std::vector<Real> build_rt_direct_transform(
    ElementType type,
    int order,
    std::size_t n,
    const std::vector<std::array<int, 4>>& extra_monomials) {
    FE_CHECK_ARG(is_wedge(type) || is_pyramid(type),
                 "build_rt_direct_transform: only wedge/pyramid RT is supported");

    const int k = order;
    const elements::ReferenceElement ref = elements::ReferenceElement::create(type);
    const bool use_seed_basis = (order == 1 || order == 2);
    const std::size_t seed_cols = use_seed_basis ? n : 0u;
    const std::size_t cols = seed_cols + extra_monomials.size();
    FE_CHECK_ARG(cols >= n,
                 "build_rt_direct_transform: candidate space is smaller than the RT DOF space");
    std::vector<Real> A(n * cols, Real(0));
    std::size_t row = 0;

    const LagrangeBasis tri_face_basis(ElementType::Triangle3, k);
    const LagrangeBasis quad_face_basis(ElementType::Quad4, k);
    const auto tri_quad = quadrature::QuadratureFactory::create(
        ElementType::Triangle3, std::max(8, 2 * k + 6), QuadratureType::GaussLegendre, /*use_cache=*/true);
    const auto quad_quad = quadrature::QuadratureFactory::create(
        ElementType::Quad4, std::max(8, 2 * k + 6), QuadratureType::GaussLegendre, /*use_cache=*/true);

    int max_extra_px = 0;
    int max_extra_py = 0;
    int max_extra_pz = 0;
    for (const auto& extra : extra_monomials) {
        max_extra_px = std::max(max_extra_px, extra[1]);
        max_extra_py = std::max(max_extra_py, extra[2]);
        max_extra_pz = std::max(max_extra_pz, extra[3]);
    }

    std::vector<Real> extra_power_x;
    std::vector<Real> extra_power_y;
    std::vector<Real> extra_power_z;
    auto eval_extra_monomials = [&](const Vec3& xi, std::vector<Real>& values) {
        values.resize(extra_monomials.size(), Real(0));
        if (extra_monomials.empty()) {
            return;
        }

        fill_powers(xi[0], max_extra_px, extra_power_x);
        fill_powers(xi[1], max_extra_py, extra_power_y);
        fill_powers(xi[2], max_extra_pz, extra_power_z);
        for (std::size_t m = 0; m < extra_monomials.size(); ++m) {
            const auto& mono = extra_monomials[m];
            values[m] =
                extra_power_x[static_cast<std::size_t>(mono[1])] *
                extra_power_y[static_cast<std::size_t>(mono[2])] *
                extra_power_z[static_cast<std::size_t>(mono[3])];
        }
    };

    std::vector<Real> face_vals;
    std::vector<Real> extra_values;
    std::vector<Vec3> seed_values;

    for (std::size_t f = 0; f < ref.num_faces(); ++f) {
        const auto& fn = ref.face_nodes(f);
        if (fn.size() == 3u) {
            const Vec3 v0 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[0]));
            const Vec3 v1 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[1]));
            const Vec3 v2 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[2]));
            const Vec3 e01 = v1 - v0;
            const Vec3 e02 = v2 - v0;
            const Vec3 cross = cross3(e01, e02);
            const std::size_t nface = tri_face_basis.size();

            FE_CHECK_ARG(row + nface <= n, "build_rt_direct_transform: triangular face row overflow");
            for (std::size_t q = 0; q < tri_quad->num_points(); ++q) {
                const auto uv = tri_quad->point(q);
                const Real u = uv[0];
                const Real v = uv[1];
                const Vec3 xi = v0 + e01 * u + e02 * v;

                face_vals.clear();
                tri_face_basis.evaluate_values(Vec3{u, v, Real(0)}, face_vals);
                extra_values.clear();
                seed_values.clear();
                if (use_seed_basis) {
                    eval_rt_seed_values(type, order, xi, seed_values);
                    FE_CHECK_ARG(seed_values.size() == n,
                                 "build_rt_direct_transform: triangular face seed basis size mismatch");
                }
                eval_extra_monomials(xi, extra_values);

                const Real wt = tri_quad->weight(q);
                if (use_seed_basis) {
                    for (std::size_t p = 0; p < n; ++p) {
                        const Real flux =
                            seed_values[p][0] * cross[0] +
                            seed_values[p][1] * cross[1] +
                            seed_values[p][2] * cross[2];
                        for (std::size_t a = 0; a < nface; ++a) {
                            A[(row + a) * cols + p] += wt * face_vals[a] * flux;
                        }
                    }
                }
                for (std::size_t m = 0; m < extra_monomials.size(); ++m) {
                    const auto& mono = extra_monomials[m];
                    const Real flux = cross[static_cast<std::size_t>(mono[0])] * extra_values[m];
                    for (std::size_t a = 0; a < nface; ++a) {
                        A[(row + a) * cols + (seed_cols + m)] += wt * face_vals[a] * flux;
                    }
                }
            }
            row += nface;
            continue;
        }

        const std::array<Vec3, 4> fv{
            ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[0])),
            ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[1])),
            ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[2])),
            ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[3]))
        };
        const std::size_t nface = quad_face_basis.size();

        FE_CHECK_ARG(row + nface <= n, "build_rt_direct_transform: quadrilateral face row overflow");
        for (std::size_t q = 0; q < quad_quad->num_points(); ++q) {
            const auto uv = quad_quad->point(q);
            const Real u = uv[0];
            const Real w = uv[1];
            const Vec3 xi = bilinear(fv, u, w);
            const Vec3 du = bilinear_du(fv, u, w);
            const Vec3 dw = bilinear_dw(fv, u, w);
            const Vec3 cross = cross3(du, dw);

            face_vals.clear();
            quad_face_basis.evaluate_values(Vec3{u, w, Real(0)}, face_vals);
            extra_values.clear();
            seed_values.clear();
            if (use_seed_basis) {
                eval_rt_seed_values(type, order, xi, seed_values);
                FE_CHECK_ARG(seed_values.size() == n,
                             "build_rt_direct_transform: quadrilateral face seed basis size mismatch");
            }
            eval_extra_monomials(xi, extra_values);

            const Real wt = quad_quad->weight(q);
            if (use_seed_basis) {
                for (std::size_t p = 0; p < n; ++p) {
                    const Real flux =
                        seed_values[p][0] * cross[0] +
                        seed_values[p][1] * cross[1] +
                        seed_values[p][2] * cross[2];
                    for (std::size_t a = 0; a < nface; ++a) {
                        A[(row + a) * cols + p] += wt * face_vals[a] * flux;
                    }
                }
            }
            for (std::size_t m = 0; m < extra_monomials.size(); ++m) {
                const auto& mono = extra_monomials[m];
                const Real flux = cross[static_cast<std::size_t>(mono[0])] * extra_values[m];
                for (std::size_t a = 0; a < nface; ++a) {
                    A[(row + a) * cols + (seed_cols + m)] += wt * face_vals[a] * flux;
                }
            }
        }
        row += nface;
    }

    const auto vol_quad = quadrature::QuadratureFactory::create(
        type,
        is_wedge(type) ? std::max(8, 2 * k + 6) : std::max(10, 2 * k + 8),
        QuadratureType::GaussLegendre,
        /*use_cache=*/true);

    std::vector<Real> test_power_x;
    std::vector<Real> test_power_y;
    std::vector<Real> test_power_z;
    auto monomial_value = [&](const Vec3& xi, int px, int py, int pz) {
        fill_powers(xi[0], px, test_power_x);
        fill_powers(xi[1], py, test_power_y);
        fill_powers(xi[2], pz, test_power_z);
        return test_power_x[static_cast<std::size_t>(px)] *
               test_power_y[static_cast<std::size_t>(py)] *
               test_power_z[static_cast<std::size_t>(pz)];
    };

    if (is_wedge(type)) {
        FE_CHECK_ARG(n == vector_common::rt_wedge_size(order),
                     "build_rt_direct_transform: unexpected wedge RT size");
        for (int c = 0; c < 3; ++c) {
            for (int l = 0; l <= k; ++l) {
                for (int j = 0; j <= k - 1; ++j) {
                    for (int i = 0; i <= k - 1 - j; ++i) {
                        FE_CHECK_ARG(row < n, "build_rt_direct_transform: wedge interior row overflow");
                        for (std::size_t q = 0; q < vol_quad->num_points(); ++q) {
                            const Vec3 xi = vol_quad->point(q);
                            const Real wt = vol_quad->weight(q);

                            extra_values.clear();
                            seed_values.clear();
                            if (use_seed_basis) {
                                eval_rt_seed_values(type, order, xi, seed_values);
                                FE_CHECK_ARG(seed_values.size() == n,
                                             "build_rt_direct_transform: wedge interior seed basis size mismatch");
                            }
                            eval_extra_monomials(xi, extra_values);

                            const Real test = monomial_value(xi, i, j, l);
                            if (use_seed_basis) {
                                for (std::size_t p = 0; p < n; ++p) {
                                    A[row * cols + p] +=
                                        wt * seed_values[p][static_cast<std::size_t>(c)] * test;
                                }
                            }
                            for (std::size_t m = 0; m < extra_monomials.size(); ++m) {
                                if (extra_monomials[m][0] != c) {
                                    continue;
                                }
                                A[row * cols + (seed_cols + m)] += wt * extra_values[m] * test;
                            }
                        }
                        ++row;
                    }
                }
            }
        }
    } else {
        FE_CHECK_ARG(n == vector_common::rt_pyramid_size(order),
                     "build_rt_direct_transform: unexpected pyramid RT size");
        for (int c = 0; c < 3; ++c) {
            for (int l = 0; l <= k - 1; ++l) {
                for (int j = 0; j <= k - 1; ++j) {
                    for (int i = 0; i <= k - 1; ++i) {
                        FE_CHECK_ARG(row < n,
                                     "build_rt_direct_transform: pyramid interior row overflow");
                        for (std::size_t q = 0; q < vol_quad->num_points(); ++q) {
                            const Vec3 xi = vol_quad->point(q);
                            const Real wt = vol_quad->weight(q);

                            extra_values.clear();
                            seed_values.clear();
                            if (use_seed_basis) {
                                eval_rt_seed_values(type, order, xi, seed_values);
                                FE_CHECK_ARG(seed_values.size() == n,
                                             "build_rt_direct_transform: pyramid interior seed basis size mismatch");
                            }
                            eval_extra_monomials(xi, extra_values);

                            const Real test = monomial_value(xi, i, j, l);
                            if (use_seed_basis) {
                                for (std::size_t p = 0; p < n; ++p) {
                                    A[row * cols + p] +=
                                        wt * seed_values[p][static_cast<std::size_t>(c)] * test;
                                }
                            }
                            for (std::size_t m = 0; m < extra_monomials.size(); ++m) {
                                if (extra_monomials[m][0] != c) {
                                    continue;
                                }
                                A[row * cols + (seed_cols + m)] += wt * extra_values[m] * test;
                            }
                        }
                        ++row;
                    }
                }
            }
        }
    }

    FE_CHECK_ARG(row == n, "build_rt_direct_transform: DOF assembly did not fill matrix");
    if (cols == n) {
        const int rank = compute_rank(A, n);
        FE_CHECK_ARG(rank == static_cast<int>(n),
                     "build_rt_direct_transform: RT seed moment matrix is rank-deficient "
                     "(rank " + std::to_string(rank) + " of " + std::to_string(n) + ")");
        return invert_dense_matrix(std::move(A), n);
    }

    return right_inverse_full_row_rank(A, n, cols);
}

void eval_nd_seed_values(ElementType type,
                         int order,
                         const Vec3& xi,
                         std::vector<Vec3>& values) {
    FE_CHECK_ARG(order == 1 || order == 2,
                 "eval_nd_seed_values: only wedge/pyramid ND(1-2) seeds are supported");
    if (is_wedge(type)) {
        if (order == 1) {
            vector_direct::eval_wedge_nd1_values(xi, values);
        } else {
            vector_direct::eval_wedge_nd2_values(xi, values);
        }
    } else {
        if (order == 1) {
            vector_direct::eval_pyramid_nd1_values(xi, values);
        } else {
            vector_direct::eval_pyramid_nd2_values(xi, values);
        }
    }
}

void eval_nd_seed_curl(ElementType type,
                       int order,
                       const Vec3& xi,
                       std::vector<Vec3>& curl) {
    FE_CHECK_ARG(order == 1 || order == 2,
                 "eval_nd_seed_curl: only wedge/pyramid ND(1-2) seeds are supported");
    if (is_wedge(type)) {
        if (order == 1) {
            vector_direct::eval_wedge_nd1_curl(xi, curl);
        } else {
            vector_direct::eval_wedge_nd2_curl(xi, curl);
        }
    } else {
        if (order == 1) {
            vector_direct::eval_pyramid_nd1_curl(xi, curl);
        } else {
            vector_direct::eval_pyramid_nd2_curl(xi, curl);
        }
    }
}

std::vector<std::array<int, 4>> make_nd_interior_tests(std::size_t count) {
    std::vector<std::array<int, 4>> tests;
    int total = 0;
    while (tests.size() < count) {
        for (int component = 0; component < 3 && tests.size() < count; ++component) {
            for (int pz = 0; pz <= total && tests.size() < count; ++pz) {
                for (int py = 0; py <= total - pz && tests.size() < count; ++py) {
                    const int px = total - py - pz;
                    tests.push_back({component, px, py, pz});
                }
            }
        }
        ++total;
    }
    return tests;
}

std::vector<Real> build_nd_direct_transform(ElementType type,
                                            int order,
                                            std::size_t n,
                                            const std::vector<std::array<int, 4>>& extra_monomials) {
    FE_CHECK_ARG(is_wedge(type) || is_pyramid(type),
                 "build_nd_direct_transform: only wedge/pyramid ND is supported");

    const int k = order;
    const elements::ReferenceElement ref = elements::ReferenceElement::create(type);
    const bool use_seed_basis = (order == 1 || order == 2);
    const std::size_t seed_cols = use_seed_basis ? n : 0u;
    const std::size_t cols = seed_cols + extra_monomials.size();
    FE_CHECK_ARG(cols >= n,
                 "build_nd_direct_transform: candidate space is smaller than the ND DOF space");
    std::vector<Real> A(n * cols, Real(0));
    std::size_t row = 0;

    int max_extra_px = 0;
    int max_extra_py = 0;
    int max_extra_pz = 0;
    for (const auto& mono : extra_monomials) {
        max_extra_px = std::max(max_extra_px, mono[1]);
        max_extra_py = std::max(max_extra_py, mono[2]);
        max_extra_pz = std::max(max_extra_pz, mono[3]);
    }

    std::vector<Real> extra_power_x;
    std::vector<Real> extra_power_y;
    std::vector<Real> extra_power_z;
    auto eval_extra_monomials = [&](const Vec3& xi, std::vector<Real>& values) {
        values.resize(extra_monomials.size(), Real(0));
        if (extra_monomials.empty()) {
            return;
        }

        fill_powers(xi[0], max_extra_px, extra_power_x);
        fill_powers(xi[1], max_extra_py, extra_power_y);
        fill_powers(xi[2], max_extra_pz, extra_power_z);
        for (std::size_t m = 0; m < extra_monomials.size(); ++m) {
            values[m] = eval_transformed_nd_monomial_scalar(
                extra_monomials[m], extra_power_x, extra_power_y, extra_power_z);
        }
    };

    std::vector<Real> lvals;
    std::vector<Real> face_vals;
    std::vector<Real> u_low_vals;
    std::vector<Real> u_full_vals;
    std::vector<Real> w_low_vals;
    std::vector<Real> w_full_vals;
    std::vector<Real> extra_values;
    std::vector<Vec3> seed_values;

    // Edge tangential moments: ∫_e (v·t) * l_i(s) ds.
    const LagrangeBasis edge_basis(ElementType::Line2, k);
    const auto edge_quad = quadrature::QuadratureFactory::create(
        ElementType::Line2, std::max(8, 2 * k + 4), QuadratureType::GaussLegendre, /*use_cache=*/true);
    for (std::size_t e = 0; e < ref.num_edges(); ++e) {
        const auto& en = ref.edge_nodes(e);
        FE_CHECK_ARG(en.size() == 2u, "build_nd_direct_transform: expected 2 nodes per edge");
        const Vec3 p0 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(en[0]));
        const Vec3 p1 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(en[1]));
        const Vec3 t = normalize3(p1 - p0);
        const Real J = math::norm(p1 - p0) * Real(0.5);

        for (int a = 0; a <= k; ++a) {
            FE_CHECK_ARG(row < n, "build_nd_direct_transform: edge row overflow");
            for (std::size_t q = 0; q < edge_quad->num_points(); ++q) {
                const Real s = edge_quad->point(q)[0];
                lvals.clear();
                edge_basis.evaluate_values(Vec3{s, Real(0), Real(0)}, lvals);
                FE_CHECK_ARG(lvals.size() == static_cast<std::size_t>(k + 1),
                             "build_nd_direct_transform: edge basis size mismatch");
                const Vec3 xi = lerp(p0, p1, s);

                extra_values.clear();
                seed_values.clear();
                if (use_seed_basis) {
                    eval_nd_seed_values(type, order, xi, seed_values);
                    FE_CHECK_ARG(seed_values.size() == n,
                                 "build_nd_direct_transform: edge seed basis size mismatch");
                }
                eval_extra_monomials(xi, extra_values);

                const Real wt = edge_quad->weight(q) * J * lvals[static_cast<std::size_t>(a)];
                if (use_seed_basis) {
                    for (std::size_t p = 0; p < n; ++p) {
                        A[row * cols + p] += wt * seed_values[p].dot(t);
                    }
                }
                for (std::size_t m = 0; m < extra_monomials.size(); ++m) {
                    const int component = extra_monomials[m][0];
                    A[row * cols + (seed_cols + m)] +=
                        wt * t[static_cast<std::size_t>(component)] * extra_values[m];
                }
            }
            ++row;
        }
    }

    if (k >= 1) {
        const auto tri_quad = quadrature::QuadratureFactory::create(
            ElementType::Triangle3, std::max(8, 2 * k + 4), QuadratureType::GaussLegendre, /*use_cache=*/true);
        const auto quad_quad = quadrature::QuadratureFactory::create(
            ElementType::Quad4, std::max(8, 2 * k + 4), QuadratureType::GaussLegendre, /*use_cache=*/true);
        const LagrangeBasis tri_face_basis(ElementType::Triangle3, k - 1);
        const LagrangeBasis u_low(ElementType::Line2, k - 1);
        const LagrangeBasis u_full(ElementType::Line2, k);
        const LagrangeBasis w_low(ElementType::Line2, k - 1);
        const LagrangeBasis w_full(ElementType::Line2, k);

        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const auto& fn = ref.face_nodes(f);
            if (fn.size() == 3u) {
                const Vec3 v0 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[0]));
                const Vec3 v1 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[1]));
                const Vec3 v2 = ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(fn[2]));
                const Vec3 tu = v1 - v0;
                const Vec3 tv = v2 - v0;
                const Real scale = cross3(tu, tv).norm();
                const std::size_t n_face = tri_face_basis.size();

                FE_CHECK_ARG(row + 2u * n_face <= n,
                             "build_nd_direct_transform: triangular face row overflow");
                const std::size_t row_u = row;
                const std::size_t row_v = row + n_face;

                for (std::size_t q = 0; q < tri_quad->num_points(); ++q) {
                    const auto uv = tri_quad->point(q);
                    const Real u = uv[0];
                    const Real v = uv[1];
                    const Vec3 xi = v0 + tu * u + tv * v;

                    face_vals.clear();
                    tri_face_basis.evaluate_values(Vec3{u, v, Real(0)}, face_vals);
                    FE_CHECK_ARG(face_vals.size() == n_face,
                                 "build_nd_direct_transform: triangular face basis size mismatch");

                    extra_values.clear();
                    seed_values.clear();
                    if (use_seed_basis) {
                        eval_nd_seed_values(type, order, xi, seed_values);
                        FE_CHECK_ARG(seed_values.size() == n,
                                     "build_nd_direct_transform: triangular face seed basis size mismatch");
                    }
                    eval_extra_monomials(xi, extra_values);

                    const Real wt = tri_quad->weight(q) * scale;
                    for (std::size_t a = 0; a < n_face; ++a) {
                        const Real wa = wt * face_vals[a];
                        if (use_seed_basis) {
                            for (std::size_t p = 0; p < n; ++p) {
                                A[(row_u + a) * cols + p] += wa * seed_values[p].dot(tu);
                                A[(row_v + a) * cols + p] += wa * seed_values[p].dot(tv);
                            }
                        }
                        for (std::size_t m = 0; m < extra_monomials.size(); ++m) {
                            const int component = extra_monomials[m][0];
                            A[(row_u + a) * cols + (seed_cols + m)] +=
                                wa * tu[static_cast<std::size_t>(component)] * extra_values[m];
                            A[(row_v + a) * cols + (seed_cols + m)] +=
                                wa * tv[static_cast<std::size_t>(component)] * extra_values[m];
                        }
                    }
                }

                row += 2u * n_face;
                continue;
            }

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

            FE_CHECK_ARG(row + n_u + n_w <= n,
                         "build_nd_direct_transform: quadrilateral face row overflow");
            const std::size_t row_u = row;
            const std::size_t row_w = row + n_u;

            for (std::size_t q = 0; q < quad_quad->num_points(); ++q) {
                const auto uv = quad_quad->point(q);
                const Real u = uv[0];
                const Real w = uv[1];
                const Vec3 xi = bilinear(fv, u, w);
                const Vec3 du = bilinear_du(fv, u, w);
                const Vec3 dw = bilinear_dw(fv, u, w);
                const Real scale = cross3(du, dw).norm();

                u_low_vals.clear();
                u_full_vals.clear();
                w_low_vals.clear();
                w_full_vals.clear();
                u_low.evaluate_values(Vec3{u, Real(0), Real(0)}, u_low_vals);
                u_full.evaluate_values(Vec3{u, Real(0), Real(0)}, u_full_vals);
                w_low.evaluate_values(Vec3{w, Real(0), Real(0)}, w_low_vals);
                w_full.evaluate_values(Vec3{w, Real(0), Real(0)}, w_full_vals);
                FE_CHECK_ARG(u_low_vals.size() == static_cast<std::size_t>(k),
                             "build_nd_direct_transform: u_low size mismatch");
                FE_CHECK_ARG(u_full_vals.size() == static_cast<std::size_t>(k + 1),
                             "build_nd_direct_transform: u_full size mismatch");
                FE_CHECK_ARG(w_low_vals.size() == static_cast<std::size_t>(k),
                             "build_nd_direct_transform: w_low size mismatch");
                FE_CHECK_ARG(w_full_vals.size() == static_cast<std::size_t>(k + 1),
                             "build_nd_direct_transform: w_full size mismatch");

                extra_values.clear();
                seed_values.clear();
                if (use_seed_basis) {
                    eval_nd_seed_values(type, order, xi, seed_values);
                    FE_CHECK_ARG(seed_values.size() == n,
                                 "build_nd_direct_transform: quadrilateral face seed basis size mismatch");
                }
                eval_extra_monomials(xi, extra_values);

                const Real wt = quad_quad->weight(q) * scale;
                for (int jw = 0; jw <= k; ++jw) {
                    for (int iu = 0; iu <= k - 1; ++iu) {
                        const std::size_t a =
                            static_cast<std::size_t>(jw) * static_cast<std::size_t>(k) +
                            static_cast<std::size_t>(iu);
                        const Real wa =
                            wt * u_low_vals[static_cast<std::size_t>(iu)] *
                            w_full_vals[static_cast<std::size_t>(jw)];
                        if (use_seed_basis) {
                            for (std::size_t p = 0; p < n; ++p) {
                                A[(row_u + a) * cols + p] += wa * seed_values[p].dot(tu);
                            }
                        }
                        for (std::size_t m = 0; m < extra_monomials.size(); ++m) {
                            const int component = extra_monomials[m][0];
                            A[(row_u + a) * cols + (seed_cols + m)] +=
                                wa * tu[static_cast<std::size_t>(component)] * extra_values[m];
                        }
                    }
                }

                for (int jw = 0; jw <= k - 1; ++jw) {
                    for (int iu = 0; iu <= k; ++iu) {
                        const std::size_t a =
                            static_cast<std::size_t>(jw) * static_cast<std::size_t>(k + 1) +
                            static_cast<std::size_t>(iu);
                        const Real wa =
                            wt * u_full_vals[static_cast<std::size_t>(iu)] *
                            w_low_vals[static_cast<std::size_t>(jw)];
                        if (use_seed_basis) {
                            for (std::size_t p = 0; p < n; ++p) {
                                A[(row_w + a) * cols + p] += wa * seed_values[p].dot(tw);
                            }
                        }
                        for (std::size_t m = 0; m < extra_monomials.size(); ++m) {
                            const int component = extra_monomials[m][0];
                            A[(row_w + a) * cols + (seed_cols + m)] +=
                                wa * tw[static_cast<std::size_t>(component)] * extra_values[m];
                        }
                    }
                }
            }

            row += n_u + n_w;
        }
    }

    if (row < n) {
        const auto vol_quad = quadrature::QuadratureFactory::create(
            type,
            is_wedge(type) ? std::max(8, 2 * k + 4) : std::max(10, 2 * k + 6),
            QuadratureType::GaussLegendre,
            /*use_cache=*/true);

        std::vector<Real> test_power_x;
        std::vector<Real> test_power_y;
        std::vector<Real> test_power_z;
        auto monomial_value = [&](const Vec3& xi, int px, int py, int pz) {
            fill_powers(xi[0], px, test_power_x);
            fill_powers(xi[1], py, test_power_y);
            fill_powers(xi[2], pz, test_power_z);
            return test_power_x[static_cast<std::size_t>(px)] *
                   test_power_y[static_cast<std::size_t>(py)] *
                   test_power_z[static_cast<std::size_t>(pz)];
        };

        const auto interior_tests = make_nd_interior_tests(n - row);
        for (const auto& test_mono : interior_tests) {
            FE_CHECK_ARG(row < n,
                         "build_nd_direct_transform: interior row overflow");
            for (std::size_t q = 0; q < vol_quad->num_points(); ++q) {
                const Vec3 xi = vol_quad->point(q);
                const Real wt = vol_quad->weight(q);

                extra_values.clear();
                seed_values.clear();
                if (use_seed_basis) {
                    eval_nd_seed_values(type, order, xi, seed_values);
                    FE_CHECK_ARG(seed_values.size() == n,
                                 "build_nd_direct_transform: interior seed size mismatch");
                }
                eval_extra_monomials(xi, extra_values);

                const Real test = monomial_value(xi, test_mono[1], test_mono[2], test_mono[3]);
                const std::size_t component = static_cast<std::size_t>(test_mono[0]);
                if (use_seed_basis) {
                    for (std::size_t p = 0; p < n; ++p) {
                        A[row * cols + p] += wt * seed_values[p][component] * test;
                    }
                }
                for (std::size_t m = 0; m < extra_monomials.size(); ++m) {
                    if (extra_monomials[m][0] != static_cast<int>(component)) {
                        continue;
                    }
                    A[row * cols + (seed_cols + m)] += wt * extra_values[m] * test;
                }
            }
            ++row;
        }
    }

    FE_CHECK_ARG(row == n, "build_nd_direct_transform: DOF assembly did not fill matrix");
    if (cols == n) {
        const int rank = compute_rank(A, n);
        FE_CHECK_ARG(rank == static_cast<int>(n),
                     "build_nd_direct_transform: ND seed moment matrix is rank-deficient "
                     "(rank " + std::to_string(rank) + " of " + std::to_string(n) + ")");
        return invert_dense_matrix(std::move(A), n);
    }
    return right_inverse_full_row_rank(A, n, cols);
}

} // namespace vector_construction
} // namespace detail
} // namespace basis
} // namespace FE
} // namespace svmp
