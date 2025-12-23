/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/OrientationManager.h"
#include "Basis/LagrangeBasis.h"
#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace svmp {
namespace FE {
namespace spaces {

namespace {

using Vec3 = math::Vector<Real, 3>;

struct Affine2 {
    Real a00{Real(1)};
    Real a01{Real(0)};
    Real a10{Real(0)};
    Real a11{Real(1)};
    Real b0{Real(0)};
    Real b1{Real(0)};

    [[nodiscard]] Vec3 apply(const Vec3& p) const noexcept {
        Vec3 out{};
        out[0] = a00 * p[0] + a01 * p[1] + b0;
        out[1] = a10 * p[0] + a11 * p[1] + b1;
        out[2] = p[2];
        return out;
    }
};

Affine2 compute_affine_from_vertex_map(const std::vector<Vec3>& verts,
                                       const std::vector<int>& local_to_global) {
    const std::size_t n = verts.size();
    FE_CHECK_ARG(n == local_to_global.size(),
                 "compute_affine_from_vertex_map: size mismatch");
    FE_CHECK_ARG(n == 3u || n == 4u,
                 "compute_affine_from_vertex_map: supported for tri/quad only");

    const auto S0 = verts[0];
    const auto S1 = verts[1];
    const auto S2 = verts[2];
    const auto T0 = verts[static_cast<std::size_t>(local_to_global[0])];
    const auto T1 = verts[static_cast<std::size_t>(local_to_global[1])];
    const auto T2 = verts[static_cast<std::size_t>(local_to_global[2])];

    const Real b00 = S1[0] - S0[0];
    const Real b01 = S2[0] - S0[0];
    const Real b10 = S1[1] - S0[1];
    const Real b11 = S2[1] - S0[1];

    const Real c00 = T1[0] - T0[0];
    const Real c01 = T2[0] - T0[0];
    const Real c10 = T1[1] - T0[1];
    const Real c11 = T2[1] - T0[1];

    const Real detB = b00 * b11 - b01 * b10;
    FE_CHECK_ARG(std::abs(detB) > Real(0),
                 "compute_affine_from_vertex_map: degenerate reference vertices");
    const Real inv_detB = Real(1) / detB;

    // invB = (1/detB) * [ b11 -b01; -b10 b00 ]
    const Real invb00 =  b11 * inv_detB;
    const Real invb01 = -b01 * inv_detB;
    const Real invb10 = -b10 * inv_detB;
    const Real invb11 =  b00 * inv_detB;

    // A = C * invB
    Affine2 map;
    map.a00 = c00 * invb00 + c01 * invb10;
    map.a01 = c00 * invb01 + c01 * invb11;
    map.a10 = c10 * invb00 + c11 * invb10;
    map.a11 = c10 * invb01 + c11 * invb11;

    // b = T0 - A*S0
    map.b0 = T0[0] - (map.a00 * S0[0] + map.a01 * S0[1]);
    map.b1 = T0[1] - (map.a10 * S0[0] + map.a11 * S0[1]);

    if (n == 4u) {
        const auto S3 = verts[3];
        const auto T3 = verts[static_cast<std::size_t>(local_to_global[3])];
        const auto mapped = map.apply(S3);
        FE_CHECK_ARG(mapped.approx_equal(T3, Real(1e-12)),
                     "compute_affine_from_vertex_map: inconsistent quad vertex map");
    }

    return map;
}

std::vector<int> compute_lagrange_face_permutation(ElementType face_type,
                                                   int poly_order,
                                                   const std::vector<int>& vertex_perm) {
    FE_CHECK_ARG(poly_order >= 0, "compute_lagrange_face_permutation: negative order");
    basis::LagrangeBasis lag(face_type, poly_order);
    const auto& nodes = lag.nodes();
    const std::size_t n = nodes.size();

    if (n == 0) {
        return {};
    }

    // Local vertex index -> global vertex index (0..nverts-1) in canonical reference element.
    const auto local_to_global = OrientationManager::invert_permutation(vertex_perm);

    std::vector<Vec3> verts;
    if (face_type == ElementType::Triangle3) {
        verts = {Vec3{Real(0), Real(0), Real(0)},
                 Vec3{Real(1), Real(0), Real(0)},
                 Vec3{Real(0), Real(1), Real(0)}};
    } else if (face_type == ElementType::Quad4) {
        verts = {Vec3{Real(-1), Real(-1), Real(0)},
                 Vec3{Real(1),  Real(-1), Real(0)},
                 Vec3{Real(1),  Real(1),  Real(0)},
                 Vec3{Real(-1), Real(1),  Real(0)}};
    } else {
        throw FEException("compute_lagrange_face_permutation: unsupported face type",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }

    FE_CHECK_ARG(verts.size() == local_to_global.size(),
                 "compute_lagrange_face_permutation: vertex map size mismatch");

    const auto map = compute_affine_from_vertex_map(verts, local_to_global);

    std::vector<int> perm(n, -1); // perm[global_index] = local_index
    std::vector<bool> used(n, false);

    for (std::size_t i_local = 0; i_local < n; ++i_local) {
        const Vec3 x_global = map.apply(nodes[i_local]);
        int matched = -1;
        for (std::size_t j = 0; j < n; ++j) {
            if (used[j]) {
                continue;
            }
            if (nodes[j].approx_equal(x_global, Real(1e-12))) {
                matched = static_cast<int>(j);
                break;
            }
        }
        FE_CHECK_ARG(matched >= 0, "compute_lagrange_face_permutation: node match failed");
        used[static_cast<std::size_t>(matched)] = true;
        perm[static_cast<std::size_t>(matched)] = static_cast<int>(i_local);
    }

    return perm;
}

std::vector<Vec3> tensor_product_quad_nodes(int order_u, int order_w) {
    FE_CHECK_ARG(order_u >= 0 && order_w >= 0, "tensor_product_quad_nodes: negative order");

    basis::LagrangeBasis bu(ElementType::Line2, order_u);
    basis::LagrangeBasis bw(ElementType::Line2, order_w);
    const auto& un = bu.nodes();
    const auto& wn = bw.nodes();

    std::vector<Vec3> nodes;
    nodes.reserve(un.size() * wn.size());

    // Ordering matches tensor-product Lagrange evaluation used in vector-basis DOF blocks:
    // outer loop is the second coordinate (w), inner loop is the first (u).
    for (std::size_t jw = 0; jw < wn.size(); ++jw) {
        for (std::size_t iu = 0; iu < un.size(); ++iu) {
            nodes.push_back(Vec3{un[iu][0], wn[jw][0], Real(0)});
        }
    }
    return nodes;
}

std::vector<int> compute_node_permutation(const std::vector<Vec3>& source_nodes,
                                          const std::vector<Vec3>& target_nodes,
                                          const Affine2& map) {
    FE_CHECK_ARG(source_nodes.size() == target_nodes.size(),
                 "compute_node_permutation: size mismatch");
    const std::size_t n = target_nodes.size();
    if (n == 0) {
        return {};
    }

    std::vector<int> perm(n, -1); // perm[target_index] = source_index
    std::vector<bool> used(n, false);

    for (std::size_t i_src = 0; i_src < n; ++i_src) {
        const Vec3 x_global = map.apply(source_nodes[i_src]);
        int matched = -1;
        for (std::size_t j = 0; j < n; ++j) {
            if (used[j]) {
                continue;
            }
            if (target_nodes[j].approx_equal(x_global, Real(1e-12))) {
                matched = static_cast<int>(j);
                break;
            }
        }
        FE_CHECK_ARG(matched >= 0, "compute_node_permutation: node match failed");
        used[static_cast<std::size_t>(matched)] = true;
        perm[static_cast<std::size_t>(matched)] = static_cast<int>(i_src);
    }
    return perm;
}

} // namespace

OrientationManager::Sign OrientationManager::edge_orientation(
    int local_v0, int local_v1,
    int ref_v0, int ref_v1) {
    if (local_v0 == ref_v0 && local_v1 == ref_v1) {
        return +1;
    }
    if (local_v0 == ref_v1 && local_v1 == ref_v0) {
        return -1;
    }
    throw FEException("edge_orientation: local edge does not match reference endpoints",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

std::vector<Real> OrientationManager::orient_edge_dofs(
    const std::vector<Real>& edge_dofs,
    Sign sign,
    bool swap_vertex_dofs) {

    std::vector<Real> result(edge_dofs.size());

    if (!swap_vertex_dofs || edge_dofs.size() < 2) {
        for (std::size_t i = 0; i < edge_dofs.size(); ++i) {
            result[i] = (sign >= 0) ? edge_dofs[i] : -edge_dofs[i];
        }
        return result;
    }

    // First two DOFs are vertex-based and must be swapped if sign < 0
    if (sign >= 0) {
        result = edge_dofs;
    } else {
        result[0] = edge_dofs[1];
        result[1] = edge_dofs[0];
        for (std::size_t i = 2; i < edge_dofs.size(); ++i) {
            result[i] = -edge_dofs[i];
        }
    }

    return result;
}

std::vector<Real> OrientationManager::orient_hcurl_edge_dofs(
    const std::vector<Real>& edge_dofs,
    Sign sign) {
    if (sign >= 0 || edge_dofs.empty()) {
        return edge_dofs;
    }

    std::vector<Real> result(edge_dofs.rbegin(), edge_dofs.rend());
    for (auto& v : result) {
        v = -v;
    }
    return result;
}

OrientationManager::FaceOrientation
OrientationManager::triangle_face_orientation(const std::array<int, 3>& local,
                                              const std::array<int, 3>& global) {
    FaceOrientation orient;
    // Determine rotation/reflection that maps `local` vertex ordering into `global`.
    for (int r = 0; r < 3; ++r) {
        std::array<int, 3> cand = {
            local[static_cast<std::size_t>((0 + r) % 3)],
            local[static_cast<std::size_t>((1 + r) % 3)],
            local[static_cast<std::size_t>((2 + r) % 3)]
        };
        if (cand == global) {
            orient.rotation = r;
            orient.reflection = false;
            orient.sign = +1;
            orient.vertex_perm = { (0 + r) % 3, (1 + r) % 3, (2 + r) % 3 };
            return orient;
        }
    }

    // Reflected winding: reverse cyclic order.
    const std::array<int, 3> refl = {local[0], local[2], local[1]};
    for (int r = 0; r < 3; ++r) {
        std::array<int, 3> cand = {
            refl[static_cast<std::size_t>((0 + r) % 3)],
            refl[static_cast<std::size_t>((1 + r) % 3)],
            refl[static_cast<std::size_t>((2 + r) % 3)]
        };
        if (cand == global) {
            orient.rotation = r;
            orient.reflection = true;
            orient.sign = -1;
            orient.vertex_perm.assign(3, -1);
            for (std::size_t gi = 0; gi < 3; ++gi) {
                for (std::size_t li = 0; li < 3; ++li) {
                    if (local[li] == global[gi]) {
                        orient.vertex_perm[gi] = static_cast<int>(li);
                        break;
                    }
                }
            }
            return orient;
        }
    }

    throw FEException("triangle_face_orientation: local/global vertex sets do not match",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

OrientationManager::FaceOrientation
OrientationManager::quad_face_orientation(const std::array<int, 4>& local,
                                          const std::array<int, 4>& global) {
    FaceOrientation orient;

    auto rotate = [&](const std::array<int, 4>& v, int r) {
        return std::array<int, 4>{
            v[static_cast<std::size_t>((0 + r) % 4)],
            v[static_cast<std::size_t>((1 + r) % 4)],
            v[static_cast<std::size_t>((2 + r) % 4)],
            v[static_cast<std::size_t>((3 + r) % 4)]
        };
    };

    for (int r = 0; r < 4; ++r) {
        const auto cand = rotate(local, r);
        if (cand == global) {
            orient.rotation = r;
            orient.reflection = false;
            orient.sign = +1;
            orient.vertex_perm = { (0 + r) % 4, (1 + r) % 4, (2 + r) % 4, (3 + r) % 4 };
            return orient;
        }
    }

    const std::array<int, 4> refl = {local[0], local[3], local[2], local[1]};
    for (int r = 0; r < 4; ++r) {
        const auto cand = rotate(refl, r);
        if (cand == global) {
            orient.rotation = r;
            orient.reflection = true;
            orient.sign = -1;
            // Compute vertex_perm by matching global vertices to local indices.
            orient.vertex_perm.assign(4, -1);
            for (std::size_t gi = 0; gi < 4; ++gi) {
                for (std::size_t li = 0; li < 4; ++li) {
                    if (local[li] == global[gi]) {
                        orient.vertex_perm[gi] = static_cast<int>(li);
                        break;
                    }
                }
            }
            return orient;
        }
    }

    throw FEException("quad_face_orientation: local/global vertex sets do not match",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

OrientationManager::Sign
OrientationManager::permutation_sign(const std::vector<int>& perm) {
    const std::size_t n = perm.size();
    std::vector<bool> visited(n, false);
    int sign = +1;

    for (std::size_t i = 0; i < n; ++i) {
        if (visited[i]) {
            continue;
        }
        // Follow cycle starting at i
        std::size_t j = i;
        std::size_t cycle_len = 0;
        while (!visited[j]) {
            visited[j] = true;
            j = static_cast<std::size_t>(perm[j]);
            ++cycle_len;
        }
        if (cycle_len > 0 && (cycle_len - 1) % 2 == 1) {
            sign = -sign;
        }
    }
    return sign;
}

std::vector<Real> OrientationManager::apply_permutation(
    const std::vector<Real>& values,
    const std::vector<int>& perm) {

    FE_CHECK_ARG(values.size() == perm.size(),
                 "apply_permutation: permutation size mismatch");

    std::vector<Real> result(values.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        int src = perm[i];
        FE_CHECK_ARG(src >= 0 && src < static_cast<int>(values.size()),
                     "apply_permutation: index out of range");
        result[i] = values[static_cast<std::size_t>(src)];
    }
    return result;
}

std::vector<int> OrientationManager::compose_permutations(
    const std::vector<int>& p1,
    const std::vector<int>& p2) {

    FE_CHECK_ARG(p1.size() == p2.size(),
                 "compose_permutations: size mismatch");

    std::vector<int> result(p1.size());
    for (std::size_t i = 0; i < p1.size(); ++i) {
        int idx = p2[i];
        FE_CHECK_ARG(idx >= 0 && idx < static_cast<int>(p1.size()),
                     "compose_permutations: index out of range");
        result[i] = p1[static_cast<std::size_t>(idx)];
    }
    return result;
}

std::vector<int> OrientationManager::invert_permutation(
    const std::vector<int>& perm) {
    std::vector<int> inv(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        int j = perm[i];
        FE_CHECK_ARG(j >= 0 && j < static_cast<int>(perm.size()),
                     "invert_permutation: index out of range");
        inv[static_cast<std::size_t>(j)] = static_cast<int>(i);
    }
    return inv;
}

std::array<int, 3> OrientationManager::canonical_ordering(const std::array<int, 3>& v) {
    std::array<int, 3> idx = {0, 1, 2};
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return v[static_cast<std::size_t>(a)] < v[static_cast<std::size_t>(b)]; });
    return idx;
}

std::array<int, 4> OrientationManager::canonical_ordering(const std::array<int, 4>& v) {
    std::array<int, 4> idx = {0, 1, 2, 3};
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return v[static_cast<std::size_t>(a)] < v[static_cast<std::size_t>(b)]; });
    return idx;
}

std::vector<Real> OrientationManager::orient_triangle_face_dofs(
    const std::vector<Real>& face_dofs,
    const FaceOrientation& orientation,
    int poly_order) {

    FE_CHECK_ARG(poly_order >= 0, "orient_triangle_face_dofs: negative poly_order");

    std::vector<Real> vals(face_dofs.size());
    for (std::size_t i = 0; i < face_dofs.size(); ++i) {
        vals[i] = (orientation.sign >= 0) ? face_dofs[i] : -face_dofs[i];
    }

    std::vector<int> perm = orientation.perm;
    if (perm.empty() && !orientation.vertex_perm.empty()) {
        perm = compute_lagrange_face_permutation(ElementType::Triangle3, poly_order,
                                                 orientation.vertex_perm);
    }

    if (!perm.empty()) {
        FE_CHECK_ARG(perm.size() == vals.size(),
                     "orient_triangle_face_dofs: permutation size mismatch");
        return apply_permutation(vals, perm);
    }

    return vals;
}

std::vector<Real> OrientationManager::orient_quad_face_dofs(
    const std::vector<Real>& face_dofs,
    const FaceOrientation& orientation,
    int poly_order) {

    FE_CHECK_ARG(poly_order >= 0, "orient_quad_face_dofs: negative poly_order");

    std::vector<Real> vals(face_dofs.size());
    for (std::size_t i = 0; i < face_dofs.size(); ++i) {
        vals[i] = (orientation.sign >= 0) ? face_dofs[i] : -face_dofs[i];
    }

    std::vector<int> perm = orientation.perm;
    if (perm.empty() && !orientation.vertex_perm.empty()) {
        perm = compute_lagrange_face_permutation(ElementType::Quad4, poly_order,
                                                 orientation.vertex_perm);
    }

    if (!perm.empty()) {
        FE_CHECK_ARG(perm.size() == vals.size(),
                     "orient_quad_face_dofs: permutation size mismatch");
        return apply_permutation(vals, perm);
    }

    return vals;
}

std::vector<Real> OrientationManager::orient_hcurl_triangle_face_dofs(
    const std::vector<Real>& face_dofs,
    const FaceOrientation& orientation,
    int poly_order) {

    FE_CHECK_ARG(poly_order >= 0, "orient_hcurl_triangle_face_dofs: negative poly_order");

    const std::size_t k = static_cast<std::size_t>(poly_order);
    if (k == 0u) {
        FE_CHECK_ARG(face_dofs.empty(),
                     "orient_hcurl_triangle_face_dofs: expected empty face_dofs for k=0");
        return face_dofs;
    }

    const std::size_t block = k * (k + 1u) / 2u; // dim(P_{k-1}) on triangle
    const std::size_t expected = 2u * block;
    FE_CHECK_ARG(face_dofs.size() == expected,
                 "orient_hcurl_triangle_face_dofs: face_dofs size mismatch");

    FE_CHECK_ARG(orientation.vertex_perm.size() == 3u,
                 "orient_hcurl_triangle_face_dofs: missing triangle vertex_perm");

    // Permute scalar basis indices for P_{k-1} using nodal Lagrange nodes.
    const auto perm_phi = compute_lagrange_face_permutation(
        ElementType::Triangle3, static_cast<int>(k - 1u), orientation.vertex_perm);
    FE_CHECK_ARG(perm_phi.size() == block,
                 "orient_hcurl_triangle_face_dofs: scalar permutation size mismatch");

    const std::vector<Real> local_u(face_dofs.begin(),
                                    face_dofs.begin() + static_cast<std::ptrdiff_t>(block));
    const std::vector<Real> local_v(face_dofs.begin() + static_cast<std::ptrdiff_t>(block),
                                    face_dofs.end());

    const std::vector<Real> u_perm = apply_permutation(local_u, perm_phi);
    const std::vector<Real> v_perm = apply_permutation(local_v, perm_phi);

    // Compute the affine map induced by the vertex permutation on the reference
    // triangle (u,v) coordinates. The linear part A has det(A)=±1 and induces
    // the contravariant transform for tangential components:
    //   [v·t_U; v·t_V] = A^{-T} [v·t_u; v·t_v]
    const auto local_to_global = invert_permutation(orientation.vertex_perm);
    const std::vector<Vec3> verts = {
        Vec3{Real(0), Real(0), Real(0)},
        Vec3{Real(1), Real(0), Real(0)},
        Vec3{Real(0), Real(1), Real(0)}
    };
    FE_CHECK_ARG(verts.size() == local_to_global.size(),
                 "orient_hcurl_triangle_face_dofs: vertex map size mismatch");

    const auto map = compute_affine_from_vertex_map(verts, local_to_global);
    const Real detA = map.a00 * map.a11 - map.a01 * map.a10;
    FE_CHECK_ARG(std::abs(detA) > Real(0),
                 "orient_hcurl_triangle_face_dofs: singular face map");

    const Real inv_det = Real(1) / detA;
    const Real b00 =  map.a11 * inv_det;
    const Real b01 = -map.a10 * inv_det;
    const Real b10 = -map.a01 * inv_det;
    const Real b11 =  map.a00 * inv_det;

    std::vector<Real> out_u(block, Real(0));
    std::vector<Real> out_v(block, Real(0));
    for (std::size_t i = 0; i < block; ++i) {
        const Real a = u_perm[i];
        const Real b = v_perm[i];
        out_u[i] = b00 * a + b01 * b;
        out_v[i] = b10 * a + b11 * b;
    }

    std::vector<Real> result;
    result.reserve(expected);
    result.insert(result.end(), out_u.begin(), out_u.end());
    result.insert(result.end(), out_v.begin(), out_v.end());
    return result;
}

std::vector<Real> OrientationManager::orient_hcurl_quad_face_dofs(
    const std::vector<Real>& face_dofs,
    const FaceOrientation& orientation,
    int poly_order) {

    FE_CHECK_ARG(poly_order >= 0, "orient_hcurl_quad_face_dofs: negative poly_order");

    const std::size_t k = static_cast<std::size_t>(poly_order);
    const std::size_t block = k * (k + 1u);
    const std::size_t expected = 2u * block;

    FE_CHECK_ARG(face_dofs.size() == expected,
                 "orient_hcurl_quad_face_dofs: face_dofs size mismatch");

    if (expected == 0u) {
        return face_dofs;
    }

    FE_CHECK_ARG(orientation.vertex_perm.size() == 4u,
                 "orient_hcurl_quad_face_dofs: missing quad vertex_perm");

    const auto local_to_global = invert_permutation(orientation.vertex_perm);
    const std::vector<Vec3> verts = {
        Vec3{Real(-1), Real(-1), Real(0)},
        Vec3{Real(1),  Real(-1), Real(0)},
        Vec3{Real(1),  Real(1),  Real(0)},
        Vec3{Real(-1), Real(1),  Real(0)}
    };
    FE_CHECK_ARG(verts.size() == local_to_global.size(),
                 "orient_hcurl_quad_face_dofs: vertex map size mismatch");

    const auto map = compute_affine_from_vertex_map(verts, local_to_global);

    auto axis_and_sign = [&](Real vx, Real vy) -> std::pair<int, Sign> {
        const Real ax = std::abs(vx);
        const Real ay = std::abs(vy);
        const Real tol = Real(1e-10);
        if (ax > Real(1) - tol && ay < tol) {
            return {0, (vx >= 0 ? +1 : -1)};
        }
        if (ay > Real(1) - tol && ax < tol) {
            return {1, (vy >= 0 ? +1 : -1)};
        }
        throw FEException("orient_hcurl_quad_face_dofs: non-axis-aligned face mapping",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    };

    const auto [axis_u, sign_u] = axis_and_sign(map.a00, map.a10);
    const auto [axis_w, sign_w] = axis_and_sign(map.a01, map.a11);
    FE_CHECK_ARG(axis_u != axis_w,
                 "orient_hcurl_quad_face_dofs: invalid axis mapping");

    const bool swap_blocks = (axis_u == 1 && axis_w == 0);
    FE_CHECK_ARG((!swap_blocks && axis_u == 0 && axis_w == 1) || swap_blocks,
                 "orient_hcurl_quad_face_dofs: unexpected axis permutation");

    const auto local_u_nodes = tensor_product_quad_nodes(static_cast<int>(k) - 1, static_cast<int>(k));
    const auto local_w_nodes = tensor_product_quad_nodes(static_cast<int>(k), static_cast<int>(k) - 1);
    const auto global_u_nodes = tensor_product_quad_nodes(static_cast<int>(k) - 1, static_cast<int>(k));
    const auto global_w_nodes = tensor_product_quad_nodes(static_cast<int>(k), static_cast<int>(k) - 1);

    const std::vector<Real> src_u(face_dofs.begin(), face_dofs.begin() + static_cast<std::ptrdiff_t>(block));
    const std::vector<Real> src_w(face_dofs.begin() + static_cast<std::ptrdiff_t>(block), face_dofs.end());

    std::vector<Real> scaled_u(src_u);
    std::vector<Real> scaled_w(src_w);
    for (auto& v : scaled_u) {
        v *= static_cast<Real>(sign_u);
    }
    for (auto& v : scaled_w) {
        v *= static_cast<Real>(sign_w);
    }

    std::vector<Real> out_u;
    std::vector<Real> out_w;

    if (!swap_blocks) {
        const auto perm_u = compute_node_permutation(local_u_nodes, global_u_nodes, map);
        const auto perm_w = compute_node_permutation(local_w_nodes, global_w_nodes, map);
        out_u = apply_permutation(scaled_u, perm_u);
        out_w = apply_permutation(scaled_w, perm_w);
    } else {
        const auto perm_u_to_w = compute_node_permutation(local_u_nodes, global_w_nodes, map);
        const auto perm_w_to_u = compute_node_permutation(local_w_nodes, global_u_nodes, map);
        out_w = apply_permutation(scaled_u, perm_u_to_w);
        out_u = apply_permutation(scaled_w, perm_w_to_u);
    }

    std::vector<Real> result;
    result.reserve(expected);
    result.insert(result.end(), out_u.begin(), out_u.end());
    result.insert(result.end(), out_w.begin(), out_w.end());
    return result;
}

} // namespace spaces
} // namespace FE
} // namespace svmp
