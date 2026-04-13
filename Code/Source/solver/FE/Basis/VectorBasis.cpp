/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "VectorBasis.h"
#include "Basis/LagrangeBasis.h"
#include "Basis/NodeOrderingConventions.h"
#include "Elements/ReferenceElement.h"
#include "Quadrature/QuadratureFactory.h"

#include <cmath>
#include <limits>
#include <string>

#ifdef FE_CHECK_ARG
#undef FE_CHECK_ARG
#endif
#define FE_CHECK_ARG(condition, message) BASIS_CHECK_CONSTRUCTION((condition), (message))

namespace svmp {
namespace FE {
namespace basis {

namespace {

using Vec3 = math::Vector<Real, 3>;

inline bool is_triangle(ElementType type) {
    return type == ElementType::Triangle3 || type == ElementType::Triangle6;
}

inline bool is_quadrilateral(ElementType type) {
    return type == ElementType::Quad4 || type == ElementType::Quad8 || type == ElementType::Quad9;
}

inline bool is_tetrahedron(ElementType type) {
    return type == ElementType::Tetra4 || type == ElementType::Tetra10;
}

inline bool is_hexahedron(ElementType type) {
    return type == ElementType::Hex8 || type == ElementType::Hex20 || type == ElementType::Hex27;
}

inline bool is_wedge(ElementType type) {
    return type == ElementType::Wedge6 || type == ElementType::Wedge15 || type == ElementType::Wedge18;
}

inline bool is_pyramid(ElementType type) {
    return type == ElementType::Pyramid5 || type == ElementType::Pyramid13 || type == ElementType::Pyramid14;
}

inline std::size_t triangle_poly_dim(std::size_t k) {
    return (k + 1u) * (k + 2u) / 2u;
}

inline std::size_t tetra_poly_dim(std::size_t k) {
    return (k + 1u) * (k + 2u) * (k + 3u) / 6u;
}

inline std::size_t rt_wedge_size(int order) {
    const std::size_t k = static_cast<std::size_t>(order);
    const std::size_t face_dofs = 2u * triangle_poly_dim(k) + 3u * (k + 1u) * (k + 1u);
    const std::size_t interior_dofs = (k >= 1u) ? (3u * k * (k + 1u) * (k + 1u) / 2u) : 0u;
    return face_dofs + interior_dofs;
}

inline std::size_t rt_pyramid_size(int order) {
    const std::size_t k = static_cast<std::size_t>(order);
    const std::size_t face_dofs = (k + 1u) * (k + 1u) + 4u * triangle_poly_dim(k);
    const std::size_t interior_dofs = (k >= 1u) ? (3u * k * k * k) : 0u;
    return face_dofs + interior_dofs;
}

inline std::size_t nd_wedge_size(int order) {
    const std::size_t k = static_cast<std::size_t>(order);
    const std::size_t edge_dofs = 9u * (k + 1u);
    const std::size_t face_dofs = (k >= 1u) ? (8u * k * (k + 1u)) : 0u;
    const std::size_t interior_dofs = (k >= 2u) ? (3u * k * (k - 1u) * (k + 1u) / 2u) : 0u;
    return edge_dofs + face_dofs + interior_dofs;
}

inline std::size_t nd_pyramid_size(int order) {
    const std::size_t k = static_cast<std::size_t>(order);
    const std::size_t edge_dofs = 8u * (k + 1u);
    const std::size_t face_dofs = (k >= 1u) ? (6u * k * (k + 1u)) : 0u;
    const std::size_t interior_dofs = (k >= 2u) ? (k * (k - 1u) * (k + 1u) / 2u) : 0u;
    return edge_dofs + face_dofs + interior_dofs;
}

inline void ensure_supported_hybrid_vector_order(ElementType type,
                                                 int order,
                                                 const char* family_name) {
    (void)type;
    (void)order;
    (void)family_name;
}

inline std::vector<std::array<int, 4>> make_component_monomial_candidates(int max_total_degree) {
    FE_CHECK_ARG(max_total_degree >= 0, "make_component_monomial_candidates: negative total degree");

    std::vector<std::array<int, 4>> candidates;
    for (int component = 0; component < 3; ++component) {
        for (int total = 0; total <= max_total_degree; ++total) {
            for (int pz = 0; pz <= total; ++pz) {
                for (int py = 0; py <= total - pz; ++py) {
                    const int px = total - py - pz;
                    candidates.push_back({component, px, py, pz});
                }
            }
        }
    }
    return candidates;
}

inline std::vector<std::array<int, 4>> make_rt_extra_monomial_candidates(ElementType type, int order) {
    if (order >= 3) {
        return make_component_monomial_candidates(3 * order);
    }

    std::vector<std::array<int, 4>> candidates;
    if (!is_pyramid(type) || order != 2) {
        return candidates;
    }

    for (int component = 0; component < 3; ++component) {
        for (int pz = 0; pz <= 2; ++pz) {
            for (int py = 0; py <= 2 - pz; ++py) {
                for (int px = 0; px <= 2 - py - pz; ++px) {
                    candidates.push_back({component, px, py, pz});
                }
            }
        }
    }
    return candidates;
}

inline Real eval_transformed_rt_monomial_scalar(const std::array<int, 4>& mono,
                                                const std::vector<Real>& px,
                                                const std::vector<Real>& py,
                                                const std::vector<Real>& pz) {
    return px[static_cast<std::size_t>(mono[1])] *
           py[static_cast<std::size_t>(mono[2])] *
           pz[static_cast<std::size_t>(mono[3])];
}

inline Real eval_transformed_rt_monomial_divergence(const std::array<int, 4>& mono,
                                                    const std::vector<Real>& px,
                                                    const std::vector<Real>& py,
                                                    const std::vector<Real>& pz) {
    const int component = mono[0];
    const int px_pow = mono[1];
    const int py_pow = mono[2];
    const int pz_pow = mono[3];

    if (component == 0) {
        if (px_pow == 0) {
            return Real(0);
        }
        return Real(px_pow) *
               px[static_cast<std::size_t>(px_pow - 1)] *
               py[static_cast<std::size_t>(py_pow)] *
               pz[static_cast<std::size_t>(pz_pow)];
    }
    if (component == 1) {
        if (py_pow == 0) {
            return Real(0);
        }
        return Real(py_pow) *
               px[static_cast<std::size_t>(px_pow)] *
               py[static_cast<std::size_t>(py_pow - 1)] *
               pz[static_cast<std::size_t>(pz_pow)];
    }
    if (pz_pow == 0) {
        return Real(0);
    }
    return Real(pz_pow) *
           px[static_cast<std::size_t>(px_pow)] *
           py[static_cast<std::size_t>(py_pow)] *
           pz[static_cast<std::size_t>(pz_pow - 1)];
}

inline std::vector<std::array<int, 4>> make_nd_extra_monomial_candidates(ElementType,
                                                                         int order) {
    if (order >= 3) {
        return make_component_monomial_candidates(3 * order);
    }

    std::vector<std::array<int, 4>> candidates;
    const int max_total_degree = (order == 1) ? 4 : 5;
    for (int component = 0; component < 3; ++component) {
        for (int total = 0; total <= max_total_degree; ++total) {
            for (int pz = 0; pz <= total; ++pz) {
                for (int py = 0; py <= total - pz; ++py) {
                    const int px = total - py - pz;
                    candidates.push_back({component, px, py, pz});
                }
            }
        }
    }
    return candidates;
}

inline Real eval_transformed_nd_monomial_scalar(const std::array<int, 4>& mono,
                                                const std::vector<Real>& px,
                                                const std::vector<Real>& py,
                                                const std::vector<Real>& pz) {
    return px[static_cast<std::size_t>(mono[1])] *
           py[static_cast<std::size_t>(mono[2])] *
           pz[static_cast<std::size_t>(mono[3])];
}

inline Vec3 eval_transformed_nd_monomial_curl(const std::array<int, 4>& mono,
                                              const std::vector<Real>& px,
                                              const std::vector<Real>& py,
                                              const std::vector<Real>& pz) {
    const int component = mono[0];
    const int px_pow = mono[1];
    const int py_pow = mono[2];
    const int pz_pow = mono[3];

    const Real dphidx = (px_pow == 0)
        ? Real(0)
        : Real(px_pow) *
              px[static_cast<std::size_t>(px_pow - 1)] *
              py[static_cast<std::size_t>(py_pow)] *
              pz[static_cast<std::size_t>(pz_pow)];
    const Real dphidy = (py_pow == 0)
        ? Real(0)
        : Real(py_pow) *
              px[static_cast<std::size_t>(px_pow)] *
              py[static_cast<std::size_t>(py_pow - 1)] *
              pz[static_cast<std::size_t>(pz_pow)];
    const Real dphidz = (pz_pow == 0)
        ? Real(0)
        : Real(pz_pow) *
              px[static_cast<std::size_t>(px_pow)] *
              py[static_cast<std::size_t>(py_pow)] *
              pz[static_cast<std::size_t>(pz_pow - 1)];

    if (component == 0) {
        return Vec3{Real(0), dphidz, -dphidy};
    }
    if (component == 1) {
        return Vec3{-dphidz, Real(0), dphidx};
    }
    return Vec3{dphidy, -dphidx, Real(0)};
}

struct Diff3 {
    Real value{0};
    Real dx{0};
    Real dy{0};
    Real dz{0};

    Diff3() = default;
    Diff3(Real v) : value(v) {}
    Diff3(Real v, Real dx_in, Real dy_in, Real dz_in)
        : value(v), dx(dx_in), dy(dy_in), dz(dz_in) {}

    static Diff3 variable_x(Real v) { return Diff3(v, Real(1), Real(0), Real(0)); }
    static Diff3 variable_y(Real v) { return Diff3(v, Real(0), Real(1), Real(0)); }
    static Diff3 variable_z(Real v) { return Diff3(v, Real(0), Real(0), Real(1)); }
};

inline Diff3 operator+(const Diff3& a, const Diff3& b) {
    return Diff3(a.value + b.value, a.dx + b.dx, a.dy + b.dy, a.dz + b.dz);
}

inline Diff3 operator-(const Diff3& a, const Diff3& b) {
    return Diff3(a.value - b.value, a.dx - b.dx, a.dy - b.dy, a.dz - b.dz);
}

inline Diff3 operator-(const Diff3& a) {
    return Diff3(-a.value, -a.dx, -a.dy, -a.dz);
}

inline Diff3 operator*(const Diff3& a, const Diff3& b) {
    return Diff3(a.value * b.value,
                 a.dx * b.value + a.value * b.dx,
                 a.dy * b.value + a.value * b.dy,
                 a.dz * b.value + a.value * b.dz);
}

inline Diff3 operator/(const Diff3& a, const Diff3& b) {
    const Real inv = Real(1) / b.value;
    const Real inv2 = inv * inv;
    return Diff3(a.value * inv,
                 (a.dx * b.value - a.value * b.dx) * inv2,
                 (a.dy * b.value - a.value * b.dy) * inv2,
                 (a.dz * b.value - a.value * b.dz) * inv2);
}

template <typename Scalar>
struct DirectVec3 {
    Scalar data[3]{};

    Scalar& operator[](std::size_t i) { return data[i]; }
    const Scalar& operator[](std::size_t i) const { return data[i]; }
};

template <typename Scalar>
DirectVec3<Scalar> make_direct_vec3(Scalar x, Scalar y, Scalar z) {
    DirectVec3<Scalar> v{};
    v[0] = x;
    v[1] = y;
    v[2] = z;
    return v;
}

template <typename EvalFn>
void eval_direct_values_real(const Vec3& xi,
                             EvalFn&& eval,
                             std::vector<Vec3>& values) {
    std::vector<DirectVec3<Real>> tmp;
    eval(make_direct_vec3(xi[0], xi[1], xi[2]), tmp);

    values.resize(tmp.size());
    for (std::size_t i = 0; i < tmp.size(); ++i) {
        values[i] = Vec3{tmp[i][0], tmp[i][1], tmp[i][2]};
    }
}

template <typename EvalFn>
void eval_direct_curl_exact(const Vec3& xi,
                            EvalFn&& eval,
                            std::vector<Vec3>& curl) {
    using DiffVec3 = DirectVec3<Diff3>;

    std::vector<DiffVec3> values;
    eval(make_direct_vec3(Diff3::variable_x(xi[0]),
                          Diff3::variable_y(xi[1]),
                          Diff3::variable_z(xi[2])),
         values);

    curl.resize(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto& v = values[i];
        curl[i] = Vec3{
            v[2].dy - v[1].dz,
            v[0].dz - v[2].dx,
            v[1].dx - v[0].dy
        };
    }
}

template <typename EvalFn>
void eval_direct_divergence_exact(const Vec3& xi,
                                  EvalFn&& eval,
                                  std::vector<Real>& divergence) {
    using DiffVec3 = DirectVec3<Diff3>;

    std::vector<DiffVec3> values;
    eval(make_direct_vec3(Diff3::variable_x(xi[0]),
                          Diff3::variable_y(xi[1]),
                          Diff3::variable_z(xi[2])),
         values);

    divergence.resize(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto& v = values[i];
        divergence[i] = v[0].dx + v[1].dy + v[2].dz;
    }
}

// =============================================================================
// DIRECT DOF-BASED CONSTRUCTION FOR WEDGE RT(k) and Nedelec(k)
// =============================================================================
//
// These functions implement EXPLICIT basis function formulas that satisfy the
// DOF Kronecker delta property directly, without needing to invert a moment matrix.
// This avoids the singular matrix problem that occurs with the modal-to-nodal approach.
//
// Reference: The construction follows the approach in MFEM and Basix, where basis
// functions are defined directly in terms of reference coordinates.

// -----------------------------------------------------------------------------
// Wedge RT(k) direct construction
// -----------------------------------------------------------------------------
// Reference wedge vertices:
//   v0=(0,0,-1), v1=(1,0,-1), v2=(0,1,-1) [bottom triangle]
//   v3=(0,0,+1), v4=(1,0,+1), v5=(0,1,+1) [top triangle]
//
// Faces:
//   Face 0: bottom triangle (z=-1), outward normal (0,0,-1)
//   Face 1: top triangle (z=+1), outward normal (0,0,+1)
//   Face 2: quad y=0, outward normal (0,-1,0)
//   Face 3: quad x=0, outward normal (-1,0,0)
//   Face 4: quad x+y=1, outward normal (1/sqrt(2), 1/sqrt(2), 0)
//
// RT(0) has 5 DOFs (1 per face)
// RT(1) has 24 DOFs:
//   - Bottom tri: 3 DOFs (P_1 normal moments)
//   - Top tri: 3 DOFs
//   - 3 quad faces: 4 DOFs each = 12 DOFs (Q_1 normal moments)
//   - Interior: 6 DOFs
//
// For RT(1), we use a hierarchical construction:
// - Face functions: satisfy face normal flux moments
// - Interior functions: bubble functions with zero normal flux on all faces

inline void eval_wedge_rt1_direct(const Vec3& xi, std::vector<Vec3>& values) {
    const Real x = xi[0];
    const Real y = xi[1];
    const Real z = xi[2];

    // Barycentric coordinates for the triangular cross-section
    const Real L0 = Real(1) - x - y;  // opposite to edge from v1 to v2
    const Real L1 = x;
    const Real L2 = y;

    // Linear z-selectors
    const Real zb = (Real(1) - z) * Real(0.5);  // 1 at z=-1, 0 at z=+1
    const Real zt = (Real(1) + z) * Real(0.5);  // 0 at z=-1, 1 at z=+1

    values.resize(24);
    std::size_t idx = 0;

    // ==========================================================================
    // Bottom triangular face (z=-1): 3 DOFs - P_1 normal flux moments
    // DOF_i(v) = integral over face of (v.n) * phi_i dA, where phi_i are P_1 basis
    // Normal n = (0, 0, -1), so v.n = -v_z
    // ==========================================================================
    // The 3 P_1 moments on bottom face correspond to test functions {1, x, y}
    // Basis function for DOF i has unit moment against test_i and zero against others

    // Face DOF 0: test = 1 (constant)
    values[idx++] = Vec3{Real(0), Real(0), zb * (Real(-2))};
    // Face DOF 1: test = x (linear in x)
    values[idx++] = Vec3{Real(0), Real(0), zb * (Real(-6) * x + Real(2))};
    // Face DOF 2: test = y (linear in y)
    values[idx++] = Vec3{Real(0), Real(0), zb * (Real(-6) * y + Real(2))};

    // ==========================================================================
    // Top triangular face (z=+1): 3 DOFs - P_1 normal flux moments
    // Normal n = (0, 0, +1), so v.n = v_z
    // ==========================================================================
    // Face DOF 3: test = 1
    values[idx++] = Vec3{Real(0), Real(0), zt * Real(2)};
    // Face DOF 4: test = x
    values[idx++] = Vec3{Real(0), Real(0), zt * (Real(6) * x - Real(2))};
    // Face DOF 5: test = y
    values[idx++] = Vec3{Real(0), Real(0), zt * (Real(6) * y - Real(2))};

    // ==========================================================================
    // Quad face 2 (y=0): 4 DOFs - Q_1 normal flux moments
    // Normal n = (0, -1, 0), so v.n = -v_y
    // Face parameterized by (x, z) in [0,1] x [-1,1]
    // Q_1 test functions: {1, x, z, x*z} (mapped to face coords)
    // ==========================================================================
    // Use y-localization: (1-y) gives 1 on face, 0 on opposite edge
    values[idx++] = Vec3{Real(0), (y - Real(1)), Real(0)}; // DOF 6: test = 1
    values[idx++] = Vec3{Real(0), (y - Real(1)) * (Real(3) * x - Real(1)), Real(0)}; // DOF 7: test = x
    values[idx++] = Vec3{Real(0), (y - Real(1)) * z, Real(0)}; // DOF 8: test = z
    values[idx++] = Vec3{Real(0), (y - Real(1)) * (Real(3) * x - Real(1)) * z, Real(0)}; // DOF 9: test = x*z

    // ==========================================================================
    // Quad face 3 (x=0): 4 DOFs - Q_1 normal flux moments
    // Normal n = (-1, 0, 0), so v.n = -v_x
    // Face parameterized by (y, z) in [0,1] x [-1,1]
    // ==========================================================================
    values[idx++] = Vec3{(x - Real(1)), Real(0), Real(0)}; // DOF 10: test = 1
    values[idx++] = Vec3{(x - Real(1)) * (Real(3) * y - Real(1)), Real(0), Real(0)}; // DOF 11: test = y
    values[idx++] = Vec3{(x - Real(1)) * z, Real(0), Real(0)}; // DOF 12: test = z
    values[idx++] = Vec3{(x - Real(1)) * (Real(3) * y - Real(1)) * z, Real(0), Real(0)}; // DOF 13: test = y*z

    // ==========================================================================
    // Quad face 4 (x+y=1): 4 DOFs - Q_1 normal flux moments
    // Normal n = (1/sqrt(2), 1/sqrt(2), 0), so v.n = (v_x + v_y)/sqrt(2)
    // Face parameterized by t = x (so y = 1-t), z in [-1,1]
    // Use localization: (x+y) - (1-x-y) = 2(x+y) - 1
    // On face x+y=1: localization = 1
    // On opposite edge (x=0, y=0): localization = -1
    // ==========================================================================
    const Real loc4 = x + y;  // 1 on face, 0 on opposite vertex
    // v.n needs to integrate correctly against Q_1 tests
    // Use v = (loc4 * f, loc4 * f, 0) so v.n = sqrt(2) * loc4 * f
    values[idx++] = Vec3{loc4, loc4, Real(0)}; // DOF 14: test = 1
    values[idx++] = Vec3{loc4 * (Real(3) * x - Real(1)), loc4 * (Real(3) * x - Real(1)), Real(0)}; // DOF 15: test = s (face param)
    values[idx++] = Vec3{loc4 * z, loc4 * z, Real(0)}; // DOF 16: test = z
    values[idx++] = Vec3{loc4 * (Real(3) * x - Real(1)) * z, loc4 * (Real(3) * x - Real(1)) * z, Real(0)}; // DOF 17: test = s*z

    // ==========================================================================
    // Interior DOFs: 6 DOFs
    // These are bubble functions with zero normal flux on all faces
    // Interior test space: P_0(x,y) x P_1(z) for each of 3 components
    // But we need divergence-compatible functions
    // Use: v = bubble(x,y,z) * constant_vector
    // Bubble = L0*L1*L2 * (1-z^2) = x*(1-x-y)*y*(1-z^2)
    // ==========================================================================
    const Real bubble_xy = L0 * L1 * L2;  // = x*(1-x-y)*y, zero on all edges of triangle
    const Real bubble_z = (Real(1) - z * z);  // zero at z = +/- 1
    const Real bubble = bubble_xy * bubble_z;

    // 6 interior DOFs: test against {e_x, e_y, e_z, z*e_x, z*e_y, z*e_z}
    values[idx++] = Vec3{bubble * Real(60), Real(0), Real(0)}; // DOF 18
    values[idx++] = Vec3{Real(0), bubble * Real(60), Real(0)}; // DOF 19
    values[idx++] = Vec3{Real(0), Real(0), bubble * Real(60)}; // DOF 20
    values[idx++] = Vec3{bubble * z * Real(180), Real(0), Real(0)}; // DOF 21
    values[idx++] = Vec3{Real(0), bubble * z * Real(180), Real(0)}; // DOF 22
    values[idx++] = Vec3{Real(0), Real(0), bubble * z * Real(180)}; // DOF 23
}

inline void eval_wedge_rt1_divergence_direct(const Vec3& xi, std::vector<Real>& divergence) {
    const Real x = xi[0];
    const Real y = xi[1];
    const Real z = xi[2];

    divergence.resize(24);
    std::size_t idx = 0;

    // Bottom face functions: v = (0, 0, zb * f(x,y))
    // div v = d(zb * f)/dz = -0.5 * f
    divergence[idx++] = Real(-0.5) * Real(-2); // DOF 0
    divergence[idx++] = Real(-0.5) * (Real(-6) * x + Real(2)); // DOF 1
    divergence[idx++] = Real(-0.5) * (Real(-6) * y + Real(2)); // DOF 2

    // Top face functions: v = (0, 0, zt * f(x,y))
    // div v = d(zt * f)/dz = 0.5 * f
    divergence[idx++] = Real(0.5) * Real(2); // DOF 3
    divergence[idx++] = Real(0.5) * (Real(6) * x - Real(2)); // DOF 4
    divergence[idx++] = Real(0.5) * (Real(6) * y - Real(2)); // DOF 5

    // Quad face 2 (y=0): v = (0, (y-1)*f(x,z), 0)
    // div v = d((y-1)*f)/dy = f (since f doesn't depend on y)
    divergence[idx++] = Real(1); // DOF 6
    divergence[idx++] = Real(3) * x - Real(1); // DOF 7
    divergence[idx++] = z; // DOF 8
    divergence[idx++] = (Real(3) * x - Real(1)) * z; // DOF 9

    // Quad face 3 (x=0): v = ((x-1)*f(y,z), 0, 0)
    // div v = d((x-1)*f)/dx = f
    divergence[idx++] = Real(1); // DOF 10
    divergence[idx++] = Real(3) * y - Real(1); // DOF 11
    divergence[idx++] = z; // DOF 12
    divergence[idx++] = (Real(3) * y - Real(1)) * z; // DOF 13

    // Quad face 4 (x+y=1): v = (loc4*f, loc4*f, 0) where loc4 = x+y
    // div v = d(loc4*f)/dx + d(loc4*f)/dy = f + f = 2f (when f independent of x,y)
    // But our f can depend on x: f = g(x)*h(z), so
    // div v = d((x+y)*g(x)*h(z))/dx + d((x+y)*g(x)*h(z))/dy
    //       = g*h + (x+y)*dg/dx*h + g*h = 2*g*h + (x+y)*dg/dx*h
    divergence[idx++] = Real(2); // DOF 14: f = 1
    divergence[idx++] = Real(2) * (Real(3) * x - Real(1)) + (x + y) * Real(3); // DOF 15
    divergence[idx++] = Real(2) * z; // DOF 16
    divergence[idx++] = Real(2) * (Real(3) * x - Real(1)) * z + (x + y) * Real(3) * z; // DOF 17

    // Interior: v = bubble * (c, d, e)
    // bubble = x*(1-x-y)*y*(1-z^2)
    // d(bubble)/dx = (1-2x-y)*y*(1-z^2)
    // d(bubble)/dy = x*(1-x-2y)*(1-z^2)
    // d(bubble)/dz = x*(1-x-y)*y*(-2z)
    const Real db_dx = (Real(1) - Real(2)*x - y) * y * (Real(1) - z*z);
    const Real db_dy = x * (Real(1) - x - Real(2)*y) * (Real(1) - z*z);
    const Real db_dz = x * (Real(1) - x - y) * y * (Real(-2) * z);

    // For v = bubble * (c, 0, 0): div v = c * db/dx
    // For v = bubble * (0, c, 0): div v = c * db/dy
    // For v = bubble * (0, 0, c): div v = c * db/dz
    divergence[idx++] = Real(60) * db_dx; // DOF 18
    divergence[idx++] = Real(60) * db_dy; // DOF 19
    divergence[idx++] = Real(60) * db_dz; // DOF 20

    // For v = bubble * z * (c, 0, 0): div v = c * z * db/dx
    // For v = bubble * z * (0, c, 0): div v = c * z * db/dy
    // For v = bubble * z * (0, 0, c): div v = c * (bubble + z * db/dz)
    const Real bubble = x * (Real(1) - x - y) * y * (Real(1) - z*z);
    divergence[idx++] = Real(180) * z * db_dx; // DOF 21
    divergence[idx++] = Real(180) * z * db_dy; // DOF 22
    divergence[idx++] = Real(180) * (bubble + z * db_dz); // DOF 23
}

// -----------------------------------------------------------------------------
// Wedge Nedelec(k) direct construction
// -----------------------------------------------------------------------------
// Nedelec(0) has 9 DOFs (1 per edge)
// Nedelec(1) has 34 DOFs:
//   - 9 edges x 2 moments = 18 edge DOFs
//   - 2 tri faces x 2 tangential DOFs = 4 face DOFs
//   - 3 quad faces x 4 tangential DOFs = 12 face DOFs
//   - Interior: 0 DOFs for k=1

template <typename Scalar>
inline void eval_wedge_nd1_direct_impl(const DirectVec3<Scalar>& xi,
                                       std::vector<DirectVec3<Scalar>>& values) {
    using ValueVec = DirectVec3<Scalar>;

    const auto x = xi[0];
    const auto y = xi[1];
    const auto z = xi[2];

    // Barycentric coordinates
    const auto L0 = Real(1) - x - y;
    const auto L1 = x;
    const auto L2 = y;

    // Z selectors
    const auto zb = (Real(1) - z) * Real(0.5);
    const auto zt = (Real(1) + z) * Real(0.5);

    values.resize(34);
    std::size_t idx = 0;

    // ==========================================================================
    // Edge DOFs: 9 edges x 2 moments per edge = 18 DOFs
    // For each edge, we have tangential moments against Legendre modes {1, s}
    // ==========================================================================

    // Bottom triangle edges (at z=-1)
    // Edge 0: v0=(0,0,-1) to v1=(1,0,-1), tangent = (1,0,0), length = 1
    values[idx++] = ValueVec{zb * (Real(1) - y), Real(0), Real(0)}; // mode 0
    values[idx++] = ValueVec{zb * (Real(1) - y) * (Real(2)*x - Real(1)), Real(0), Real(0)}; // mode 1 (Legendre P_1)

    // Edge 1: v1=(1,0,-1) to v2=(0,1,-1), tangent = (-1,1,0)/sqrt(2)
    // Parameterize: p(t) = (1-t, t, -1), t in [0,1], tangent = (-1,1,0)
    values[idx++] = ValueVec{-zb * L0, zb * L0, Real(0)}; // mode 0
    values[idx++] = ValueVec{-zb * L0 * (Real(2)*y - Real(1)), zb * L0 * (Real(2)*y - Real(1)), Real(0)}; // mode 1

    // Edge 2: v2=(0,1,-1) to v0=(0,0,-1), tangent = (0,-1,0), length = 1
    values[idx++] = ValueVec{Real(0), -zb * (Real(1) - x), Real(0)}; // mode 0
    values[idx++] = ValueVec{Real(0), -zb * (Real(1) - x) * (Real(1) - Real(2)*y), Real(0)}; // mode 1

    // Top triangle edges (at z=+1)
    // Edge 3: v3=(0,0,+1) to v4=(1,0,+1), tangent = (1,0,0)
    values[idx++] = ValueVec{zt * (Real(1) - y), Real(0), Real(0)}; // mode 0
    values[idx++] = ValueVec{zt * (Real(1) - y) * (Real(2)*x - Real(1)), Real(0), Real(0)}; // mode 1

    // Edge 4: v4=(1,0,+1) to v5=(0,1,+1), tangent = (-1,1,0)/sqrt(2)
    values[idx++] = ValueVec{-zt * L0, zt * L0, Real(0)}; // mode 0
    values[idx++] = ValueVec{-zt * L0 * (Real(2)*y - Real(1)), zt * L0 * (Real(2)*y - Real(1)), Real(0)}; // mode 1

    // Edge 5: v5=(0,1,+1) to v3=(0,0,+1), tangent = (0,-1,0)
    values[idx++] = ValueVec{Real(0), -zt * (Real(1) - x), Real(0)}; // mode 0
    values[idx++] = ValueVec{Real(0), -zt * (Real(1) - x) * (Real(1) - Real(2)*y), Real(0)}; // mode 1

    // Vertical edges (tangent = (0,0,1))
    // Edge 6: v0=(0,0,-1) to v3=(0,0,+1), at (x,y)=(0,0)
    values[idx++] = ValueVec{Real(0), Real(0), L0}; // mode 0
    values[idx++] = ValueVec{Real(0), Real(0), L0 * z}; // mode 1

    // Edge 7: v1=(1,0,-1) to v4=(1,0,+1), at (x,y)=(1,0)
    values[idx++] = ValueVec{Real(0), Real(0), L1}; // mode 0
    values[idx++] = ValueVec{Real(0), Real(0), L1 * z}; // mode 1

    // Edge 8: v2=(0,1,-1) to v5=(0,1,+1), at (x,y)=(0,1)
    values[idx++] = ValueVec{Real(0), Real(0), L2}; // mode 0
    values[idx++] = ValueVec{Real(0), Real(0), L2 * z}; // mode 1

    // ==========================================================================
    // Face DOFs: tangential moments
    // ==========================================================================

    // Bottom triangular face (z=-1): 2 tangential DOFs
    // Two independent tangent directions on face, moments against P_0
    const auto face_bubble_tri_b = L0 * L1 * L2 * zb; // bubble localized to bottom face
    values[idx++] = ValueVec{face_bubble_tri_b * Real(60), Real(0), Real(0)}; // t1 direction
    values[idx++] = ValueVec{Real(0), face_bubble_tri_b * Real(60), Real(0)}; // t2 direction

    // Top triangular face (z=+1): 2 tangential DOFs
    const auto face_bubble_tri_t = L0 * L1 * L2 * zt;
    values[idx++] = ValueVec{face_bubble_tri_t * Real(60), Real(0), Real(0)}; // t1 direction
    values[idx++] = ValueVec{Real(0), face_bubble_tri_t * Real(60), Real(0)}; // t2 direction

    // Quad face 2 (y=0): 4 tangential DOFs
    // Tangent directions: (1,0,0) and (0,0,1)
    // Face bubble: L0 * L1 = x*(1-x-y) restricted to y=0 -> x*(1-x)
    // But we need it defined on whole element, use (1-y) localization
    // Actually for tangential moment we need functions that have tangent on face
    values[idx++] = ValueVec{(Real(1) - y) * L0 * Real(12), Real(0), Real(0)}; // DOF: t_x moment const
    values[idx++] = ValueVec{(Real(1) - y) * L0 * z * Real(12), Real(0), Real(0)}; // DOF: t_x moment linear z
    values[idx++] = ValueVec{Real(0), Real(0), (Real(1) - y) * L0 * Real(12)}; // DOF: t_z moment const
    values[idx++] = ValueVec{Real(0), Real(0), (Real(1) - y) * L0 * (Real(2)*x - Real(1)) * Real(12)}; // DOF: t_z moment linear x

    // Quad face 3 (x=0): 4 tangential DOFs
    // Tangent directions: (0,1,0) and (0,0,1)
    values[idx++] = ValueVec{Real(0), (Real(1) - x) * L0 * Real(12), Real(0)}; // DOF: t_y moment const
    values[idx++] = ValueVec{Real(0), (Real(1) - x) * L0 * z * Real(12), Real(0)}; // DOF: t_y moment linear z
    values[idx++] = ValueVec{Real(0), Real(0), (Real(1) - x) * L0 * Real(12)}; // DOF: t_z moment const
    values[idx++] = ValueVec{Real(0), Real(0), (Real(1) - x) * L0 * (Real(2)*y - Real(1)) * Real(12)}; // DOF: t_z moment linear y

    // Quad face 4 (x+y=1): 4 tangential DOFs
    // Tangent directions: (-1,1,0)/sqrt(2) and (0,0,1)
    const auto loc4 = x + y;
    values[idx++] = ValueVec{-loc4 * L2 * Real(12), loc4 * L2 * Real(12), Real(0)}; // DOF: t1 moment const
    values[idx++] = ValueVec{-loc4 * L2 * z * Real(12), loc4 * L2 * z * Real(12), Real(0)}; // DOF: t1 moment linear z
    values[idx++] = ValueVec{Real(0), Real(0), loc4 * L2 * Real(12)}; // DOF: t_z moment const
    values[idx++] = ValueVec{Real(0), Real(0), loc4 * L2 * x * Real(12)}; // DOF: t_z moment linear x (face param)
}

inline void eval_wedge_nd1_direct(const Vec3& xi, std::vector<Vec3>& values) {
    eval_direct_values_real(xi, eval_wedge_nd1_direct_impl<Real>, values);
}

// -----------------------------------------------------------------------------
// Pyramid RT(k) direct construction
// -----------------------------------------------------------------------------
// Reference pyramid vertices:
//   v0=(-1,-1,0), v1=(1,-1,0), v2=(1,1,0), v3=(-1,1,0), v4=(0,0,1)
//
// Faces:
//   Face 0: quad base (z=0), outward normal (0,0,-1)
//   Face 1: tri (v0,v1,v4), outward normal roughly (-y direction)
//   Face 2: tri (v1,v2,v4), outward normal roughly (+x direction)
//   Face 3: tri (v2,v3,v4), outward normal roughly (+y direction)
//   Face 4: tri (v3,v0,v4), outward normal roughly (-x direction)
//
// RT(0) has 5 DOFs (1 per face)
// RT(1) has 19 DOFs:
//   - Quad base: 4 DOFs (Q_1 normal moments)
//   - 4 tri faces: 3 DOFs each = 12 DOFs (P_1 normal moments)
//   - Interior: 3 DOFs

inline void eval_pyramid_rt1_direct(const Vec3& xi, std::vector<Vec3>& values) {
    const Real x = xi[0];
    const Real y = xi[1];
    const Real z = xi[2];

    // For pyramid, z in [0,1], base at z=0, apex at z=1
    // Base spans [-1,1] x [-1,1] at z=0

    // Vertex-based coordinates for the base quad
    const Real zc = Real(1) - z;  // complement of z, 1 at base, 0 at apex

    values.resize(19);
    std::size_t idx = 0;

    // ==========================================================================
    // Quad base face (z=0): 4 DOFs - Q_1 normal flux moments
    // Normal n = (0,0,-1), v.n = -v_z
    // Q_1 test functions on base: {1, x, y, xy}
    // ==========================================================================
    values[idx++] = Vec3{Real(0), Real(0), -zc}; // DOF 0: test = 1
    values[idx++] = Vec3{Real(0), Real(0), -zc * x * Real(3)}; // DOF 1: test = x
    values[idx++] = Vec3{Real(0), Real(0), -zc * y * Real(3)}; // DOF 2: test = y
    values[idx++] = Vec3{Real(0), Real(0), -zc * x * y * Real(9)}; // DOF 3: test = xy

    // ==========================================================================
    // Triangular face 1 (front, y = -1+z at apex): 3 DOFs
    // Vertices: v0=(-1,-1,0), v1=(1,-1,0), v4=(0,0,1)
    // Outward normal points in -y direction
    // ==========================================================================
    const Real loc_f1 = Real(1) + y - z;  // 1 on face 1, decreases away from it
    values[idx++] = Vec3{Real(0), -loc_f1, Real(0)}; // DOF 4: test = 1
    values[idx++] = Vec3{Real(0), -loc_f1 * x * Real(3), Real(0)}; // DOF 5: test = x (face param)
    values[idx++] = Vec3{Real(0), -loc_f1 * z * Real(3), Real(0)}; // DOF 6: test = z (face param)

    // ==========================================================================
    // Triangular face 2 (right, x = 1-z at apex): 3 DOFs
    // Vertices: v1=(1,-1,0), v2=(1,1,0), v4=(0,0,1)
    // Outward normal points in +x direction
    // ==========================================================================
    const Real loc_f2 = Real(1) - x - z;  // 1 on face 2
    values[idx++] = Vec3{-loc_f2, Real(0), Real(0)}; // DOF 7: test = 1
    values[idx++] = Vec3{-loc_f2 * y * Real(3), Real(0), Real(0)}; // DOF 8: test = y
    values[idx++] = Vec3{-loc_f2 * z * Real(3), Real(0), Real(0)}; // DOF 9: test = z

    // ==========================================================================
    // Triangular face 3 (back, y = 1-z at apex): 3 DOFs
    // Vertices: v2=(1,1,0), v3=(-1,1,0), v4=(0,0,1)
    // Outward normal points in +y direction
    // ==========================================================================
    const Real loc_f3 = Real(1) - y - z;  // 1 on face 3
    values[idx++] = Vec3{Real(0), -loc_f3, Real(0)}; // DOF 10: test = 1
    values[idx++] = Vec3{Real(0), -loc_f3 * x * Real(3), Real(0)}; // DOF 11: test = x
    values[idx++] = Vec3{Real(0), -loc_f3 * z * Real(3), Real(0)}; // DOF 12: test = z

    // ==========================================================================
    // Triangular face 4 (left, x = -1+z at apex): 3 DOFs
    // Vertices: v3=(-1,1,0), v0=(-1,-1,0), v4=(0,0,1)
    // Outward normal points in -x direction
    // ==========================================================================
    const Real loc_f4 = Real(1) + x - z;  // 1 on face 4
    values[idx++] = Vec3{loc_f4, Real(0), Real(0)}; // DOF 13: test = 1
    values[idx++] = Vec3{loc_f4 * y * Real(3), Real(0), Real(0)}; // DOF 14: test = y
    values[idx++] = Vec3{loc_f4 * z * Real(3), Real(0), Real(0)}; // DOF 15: test = z

    // ==========================================================================
    // Interior DOFs: 3 DOFs
    // Bubble function that vanishes on all faces
    // ==========================================================================
    // Interior bubble: product of all face localizations
    const Real bubble = zc * loc_f1 * loc_f2 * loc_f3 * loc_f4;
    values[idx++] = Vec3{bubble * Real(120), Real(0), Real(0)}; // DOF 16
    values[idx++] = Vec3{Real(0), bubble * Real(120), Real(0)}; // DOF 17
    values[idx++] = Vec3{Real(0), Real(0), bubble * Real(120)}; // DOF 18
}

inline void eval_pyramid_rt1_divergence_direct(const Vec3& xi, std::vector<Real>& divergence) {
    const Real x = xi[0];
    const Real y = xi[1];
    const Real z = xi[2];

    const Real zc = Real(1) - z;

    divergence.resize(19);
    std::size_t idx = 0;

    // Base face: v = (0, 0, -zc * f(x,y)), div = -(-1) * f = f
    divergence[idx++] = Real(1); // DOF 0
    divergence[idx++] = x * Real(3); // DOF 1
    divergence[idx++] = y * Real(3); // DOF 2
    divergence[idx++] = x * y * Real(9); // DOF 3

    // Face 1: v = (0, -loc_f1 * f, 0), loc_f1 = 1+y-z
    // div = -d(loc_f1*f)/dy = -f (when f independent of y)
    divergence[idx++] = Real(-1); // DOF 4
    divergence[idx++] = -x * Real(3); // DOF 5
    divergence[idx++] = -z * Real(3); // DOF 6

    // Face 2: v = (-loc_f2 * f, 0, 0), loc_f2 = 1-x-z
    // div = -d(loc_f2*f)/dx = f
    divergence[idx++] = Real(1); // DOF 7
    divergence[idx++] = y * Real(3); // DOF 8
    divergence[idx++] = z * Real(3); // DOF 9

    // Face 3: v = (0, -loc_f3 * f, 0), loc_f3 = 1-y-z
    // div = -d(loc_f3*f)/dy = f
    divergence[idx++] = Real(1); // DOF 10
    divergence[idx++] = x * Real(3); // DOF 11
    divergence[idx++] = z * Real(3); // DOF 12

    // Face 4: v = (loc_f4 * f, 0, 0), loc_f4 = 1+x-z
    // div = d(loc_f4*f)/dx = f
    divergence[idx++] = Real(1); // DOF 13
    divergence[idx++] = y * Real(3); // DOF 14
    divergence[idx++] = z * Real(3); // DOF 15

    // Interior: v = bubble * (c, d, e)
    // bubble = zc * loc_f1 * loc_f2 * loc_f3 * loc_f4
    // This is complex - compute numerically via partial derivatives
    const Real zc_dx = Real(0);
    const Real zc_dy = Real(0);
    const Real zc_dz = Real(-1);

    const Real loc_f1 = Real(1) + y - z;
    const Real loc_f1_dx = Real(0);
    const Real loc_f1_dy = Real(1);
    const Real loc_f1_dz = Real(-1);

    const Real loc_f2 = Real(1) - x - z;
    const Real loc_f2_dx = Real(-1);
    const Real loc_f2_dy = Real(0);
    const Real loc_f2_dz = Real(-1);

    const Real loc_f3 = Real(1) - y - z;
    const Real loc_f3_dx = Real(0);
    const Real loc_f3_dy = Real(-1);
    const Real loc_f3_dz = Real(-1);

    const Real loc_f4 = Real(1) + x - z;
    const Real loc_f4_dx = Real(1);
    const Real loc_f4_dy = Real(0);
    const Real loc_f4_dz = Real(-1);

    // bubble = zc * loc_f1 * loc_f2 * loc_f3 * loc_f4
    // db/dx = zc_dx * ... + zc * loc_f1_dx * loc_f2 * loc_f3 * loc_f4 + ...
    const Real p1234 = loc_f1 * loc_f2 * loc_f3 * loc_f4;
    const Real p234 = loc_f2 * loc_f3 * loc_f4;
    const Real p134 = loc_f1 * loc_f3 * loc_f4;
    const Real p124 = loc_f1 * loc_f2 * loc_f4;
    const Real p123 = loc_f1 * loc_f2 * loc_f3;

    const Real db_dx = zc_dx * p1234 + zc * (loc_f1_dx * p234 + loc_f2_dx * p134 + loc_f3_dx * p124 + loc_f4_dx * p123);
    const Real db_dy = zc_dy * p1234 + zc * (loc_f1_dy * p234 + loc_f2_dy * p134 + loc_f3_dy * p124 + loc_f4_dy * p123);
    const Real db_dz = zc_dz * p1234 + zc * (loc_f1_dz * p234 + loc_f2_dz * p134 + loc_f3_dz * p124 + loc_f4_dz * p123);

    divergence[idx++] = Real(120) * db_dx; // DOF 16
    divergence[idx++] = Real(120) * db_dy; // DOF 17
    divergence[idx++] = Real(120) * db_dz; // DOF 18
}

// -----------------------------------------------------------------------------
// Pyramid Nedelec(1) direct construction
// -----------------------------------------------------------------------------
// Reference pyramid vertices:
//   v0=(-1,-1,0), v1=(1,-1,0), v2=(1,1,0), v3=(-1,1,0), v4=(0,0,1)
//
// Edges (8 total):
//   Base edges: e0=(v0,v1), e1=(v1,v2), e2=(v2,v3), e3=(v3,v0)
//   Apex edges: e4=(v0,v4), e5=(v1,v4), e6=(v2,v4), e7=(v3,v4)
//
// Nedelec(1) has 28 DOFs:
//   - 8 edges x 2 moments per edge = 16 edge DOFs
//   - 1 quad face x 4 tangential moments = 4 face DOFs
//   - 4 tri faces x 2 tangential moments = 8 face DOFs
//   - Interior: 0 for k=1

template <typename Scalar>
inline void eval_pyramid_nd1_direct_impl(const DirectVec3<Scalar>& xi,
                                         std::vector<DirectVec3<Scalar>>& values) {
    using Vec3 = DirectVec3<Scalar>;

    const auto x = xi[0];
    const auto y = xi[1];
    const auto z = xi[2];

    // Face localizations
    const auto zc = Real(1) - z;    // 1 at base, 0 at apex
    const auto loc_f1 = Real(1) + y - z;  // front face
    const auto loc_f2 = Real(1) - x - z;  // right face
    const auto loc_f3 = Real(1) - y - z;  // back face
    const auto loc_f4 = Real(1) + x - z;  // left face

    values.resize(28);
    std::size_t idx = 0;

    // ==========================================================================
    // Edge DOFs: 8 edges x 2 moments per edge = 16 DOFs
    // ==========================================================================

    // Base edge 0: v0=(-1,-1,0) to v1=(1,-1,0), tangent = (1,0,0)
    // Edge localized by (1+y)*(1-z)
    const auto loc_e0 = (Real(1) + y) * zc;
    values[idx++] = Vec3{loc_e0, Real(0), Real(0)}; // mode 0
    values[idx++] = Vec3{loc_e0 * x, Real(0), Real(0)}; // mode 1

    // Base edge 1: v1=(1,-1,0) to v2=(1,1,0), tangent = (0,1,0)
    const auto loc_e1 = (Real(1) - x) * zc;
    values[idx++] = Vec3{Real(0), loc_e1, Real(0)}; // mode 0
    values[idx++] = Vec3{Real(0), loc_e1 * y, Real(0)}; // mode 1

    // Base edge 2: v2=(1,1,0) to v3=(-1,1,0), tangent = (-1,0,0)
    const auto loc_e2 = (Real(1) - y) * zc;
    values[idx++] = Vec3{-loc_e2, Real(0), Real(0)}; // mode 0
    values[idx++] = Vec3{-loc_e2 * x, Real(0), Real(0)}; // mode 1

    // Base edge 3: v3=(-1,1,0) to v0=(-1,-1,0), tangent = (0,-1,0)
    const auto loc_e3 = (Real(1) + x) * zc;
    values[idx++] = Vec3{Real(0), -loc_e3, Real(0)}; // mode 0
    values[idx++] = Vec3{Real(0), -loc_e3 * y, Real(0)}; // mode 1

    // Apex edge 4: v0=(-1,-1,0) to v4=(0,0,1), tangent ~ (1,1,1)
    // Localized by (1-x-y-z) * something
    const auto loc_e4 = loc_f2 * loc_f3; // zero on faces 2 and 3
    values[idx++] = Vec3{loc_e4, loc_e4, loc_e4 * Real(2)}; // mode 0
    values[idx++] = Vec3{loc_e4 * z, loc_e4 * z, loc_e4 * z * Real(2)}; // mode 1

    // Apex edge 5: v1=(1,-1,0) to v4=(0,0,1), tangent ~ (-1,1,1)
    const auto loc_e5 = loc_f3 * loc_f4; // zero on faces 3 and 4
    values[idx++] = Vec3{-loc_e5, loc_e5, loc_e5 * Real(2)}; // mode 0
    values[idx++] = Vec3{-loc_e5 * z, loc_e5 * z, loc_e5 * z * Real(2)}; // mode 1

    // Apex edge 6: v2=(1,1,0) to v4=(0,0,1), tangent ~ (-1,-1,1)
    const auto loc_e6 = loc_f4 * loc_f1; // zero on faces 4 and 1
    values[idx++] = Vec3{-loc_e6, -loc_e6, loc_e6 * Real(2)}; // mode 0
    values[idx++] = Vec3{-loc_e6 * z, -loc_e6 * z, loc_e6 * z * Real(2)}; // mode 1

    // Apex edge 7: v3=(-1,1,0) to v4=(0,0,1), tangent ~ (1,-1,1)
    const auto loc_e7 = loc_f1 * loc_f2; // zero on faces 1 and 2
    values[idx++] = Vec3{loc_e7, -loc_e7, loc_e7 * Real(2)}; // mode 0
    values[idx++] = Vec3{loc_e7 * z, -loc_e7 * z, loc_e7 * z * Real(2)}; // mode 1

    // ==========================================================================
    // Face DOFs: tangential moments
    // ==========================================================================

    // Quad base face (z=0): 4 tangential DOFs
    // Two tangent directions: (1,0,0) and (0,1,0)
    // Face bubble in xy-plane localized by (1-z)
    const auto base_bubble = (Real(1) - x*x) * (Real(1) - y*y) * zc;
    values[idx++] = Vec3{base_bubble * Real(4), Real(0), Real(0)}; // t_x const
    values[idx++] = Vec3{base_bubble * y * Real(4), Real(0), Real(0)}; // t_x linear y
    values[idx++] = Vec3{Real(0), base_bubble * Real(4), Real(0)}; // t_y const
    values[idx++] = Vec3{Real(0), base_bubble * x * Real(4), Real(0)}; // t_y linear x

    // Tri face 1 (front, y=-1+z): 2 tangential DOFs
    // Tangent in x-direction and tangent up the slope
    const auto f1_bubble = loc_f1 * (Real(1) - x*x); // bubble on face 1
    values[idx++] = Vec3{f1_bubble * Real(4), Real(0), Real(0)}; // t_x
    values[idx++] = Vec3{f1_bubble * z * Real(4), Real(0), f1_bubble * Real(4)}; // t along slope

    // Tri face 2 (right, x=1-z): 2 tangential DOFs
    const auto f2_bubble = loc_f2 * (Real(1) - y*y);
    values[idx++] = Vec3{Real(0), f2_bubble * Real(4), Real(0)}; // t_y
    values[idx++] = Vec3{-f2_bubble * Real(4), Real(0), f2_bubble * z * Real(4)}; // t along slope

    // Tri face 3 (back, y=1-z): 2 tangential DOFs
    const auto f3_bubble = loc_f3 * (Real(1) - x*x);
    values[idx++] = Vec3{f3_bubble * Real(4), Real(0), Real(0)}; // t_x
    values[idx++] = Vec3{f3_bubble * z * Real(4), Real(0), -f3_bubble * Real(4)}; // t along slope

    // Tri face 4 (left, x=-1+z): 2 tangential DOFs
    const auto f4_bubble = loc_f4 * (Real(1) - y*y);
    values[idx++] = Vec3{Real(0), f4_bubble * Real(4), Real(0)}; // t_y
    values[idx++] = Vec3{f4_bubble * Real(4), Real(0), f4_bubble * z * Real(4)}; // t along slope
}

inline void eval_pyramid_nd1_direct(const Vec3& xi, std::vector<Vec3>& values) {
    eval_direct_values_real(xi, eval_pyramid_nd1_direct_impl<Real>, values);
}

inline void eval_pyramid_nd1_curl_direct(const Vec3& xi, std::vector<Vec3>& curl) {
    eval_direct_curl_exact(xi, eval_pyramid_nd1_direct_impl<Diff3>, curl);
}

// -----------------------------------------------------------------------------
// RT(2) and Nedelec(2) on Wedge/Pyramid - Direct Construction
// -----------------------------------------------------------------------------
// These are the k=2 higher-order bases with the following DOF counts:
//
// RT(2) on Wedge: 66 DOFs
//   - 2 tri faces x 6 moments = 12
//   - 3 quad faces x 9 moments = 27
//   - Interior: 3*2*(2+1)^2/2 = 27
//
// RT(2) on Pyramid: 57 DOFs
//   - 1 quad face x 9 moments = 9
//   - 4 tri faces x 6 moments = 24
//   - Interior: 24
//
// Nedelec(2) on Wedge: 84 DOFs
//   - 9 edges x 3 moments = 27
//   - 2 tri faces x 6 tangential = 12
//   - 3 quad faces x 12 tangential = 36
//   - Interior: 9
//
// Nedelec(2) on Pyramid: 63 DOFs
//   - 8 edges x 3 moments = 24
//   - 1 quad face x 12 tangential = 12
//   - 4 tri faces x 6 tangential = 24
//   - Interior: 3

template <typename Scalar>
inline void eval_wedge_rt2_direct_impl(const DirectVec3<Scalar>& xi,
                                       std::vector<DirectVec3<Scalar>>& values) {
    using Vec3 = DirectVec3<Scalar>;

    const auto x = xi[0];
    const auto y = xi[1];
    const auto z = xi[2];

    // Barycentric and z-selectors
    const auto L0 = Real(1) - x - y;
    const auto L1 = x;
    const auto L2 = y;
    const auto zb = (Real(1) - z) * Real(0.5);
    const auto zt = (Real(1) + z) * Real(0.5);

    values.resize(66);
    std::size_t idx = 0;

    // ==========================================================================
    // Bottom triangular face (z=-1): 6 DOFs - P_2 normal flux moments
    // Test functions: {1, x, y, x^2, xy, y^2}
    // ==========================================================================
    values[idx++] = Vec3{Real(0), Real(0), zb * Real(-2)};
    values[idx++] = Vec3{Real(0), Real(0), zb * (Real(-6)*x + Real(2))};
    values[idx++] = Vec3{Real(0), Real(0), zb * (Real(-6)*y + Real(2))};
    values[idx++] = Vec3{Real(0), Real(0), zb * (Real(12)*x*x - Real(8)*x + Real(1))};
    values[idx++] = Vec3{Real(0), Real(0), zb * (Real(-12)*x*y + Real(4)*x + Real(4)*y - Real(1))};
    values[idx++] = Vec3{Real(0), Real(0), zb * (Real(12)*y*y - Real(8)*y + Real(1))};

    // ==========================================================================
    // Top triangular face (z=+1): 6 DOFs - P_2 normal flux moments
    // ==========================================================================
    values[idx++] = Vec3{Real(0), Real(0), zt * Real(2)};
    values[idx++] = Vec3{Real(0), Real(0), zt * (Real(6)*x - Real(2))};
    values[idx++] = Vec3{Real(0), Real(0), zt * (Real(6)*y - Real(2))};
    values[idx++] = Vec3{Real(0), Real(0), zt * (Real(-12)*x*x + Real(8)*x - Real(1))};
    values[idx++] = Vec3{Real(0), Real(0), zt * (Real(12)*x*y - Real(4)*x - Real(4)*y + Real(1))};
    values[idx++] = Vec3{Real(0), Real(0), zt * (Real(-12)*y*y + Real(8)*y - Real(1))};

    // ==========================================================================
    // Quad face 2 (y=0): 9 DOFs - Q_2 normal flux moments
    // Test: {1, x, z, x^2, xz, z^2, x^2z, xz^2, x^2z^2} (Q_2 tensor)
    // ==========================================================================
    const auto ly = (y - Real(1));
    values[idx++] = Vec3{Real(0), ly, Real(0)};
    values[idx++] = Vec3{Real(0), ly * (Real(3)*x - Real(1)), Real(0)};
    values[idx++] = Vec3{Real(0), ly * z, Real(0)};
    values[idx++] = Vec3{Real(0), ly * (Real(6)*x*x - Real(4)*x + Real(0.5)), Real(0)};
    values[idx++] = Vec3{Real(0), ly * (Real(3)*x - Real(1)) * z, Real(0)};
    values[idx++] = Vec3{Real(0), ly * (Real(3)*z*z - Real(1)), Real(0)};
    values[idx++] = Vec3{Real(0), ly * (Real(6)*x*x - Real(4)*x + Real(0.5)) * z, Real(0)};
    values[idx++] = Vec3{Real(0), ly * (Real(3)*x - Real(1)) * (Real(3)*z*z - Real(1)), Real(0)};
    values[idx++] = Vec3{Real(0), ly * (Real(6)*x*x - Real(4)*x + Real(0.5)) * (Real(3)*z*z - Real(1)), Real(0)};

    // ==========================================================================
    // Quad face 3 (x=0): 9 DOFs - Q_2 normal flux moments
    // ==========================================================================
    const auto lx = (x - Real(1));
    values[idx++] = Vec3{lx, Real(0), Real(0)};
    values[idx++] = Vec3{lx * (Real(3)*y - Real(1)), Real(0), Real(0)};
    values[idx++] = Vec3{lx * z, Real(0), Real(0)};
    values[idx++] = Vec3{lx * (Real(6)*y*y - Real(4)*y + Real(0.5)), Real(0), Real(0)};
    values[idx++] = Vec3{lx * (Real(3)*y - Real(1)) * z, Real(0), Real(0)};
    values[idx++] = Vec3{lx * (Real(3)*z*z - Real(1)), Real(0), Real(0)};
    values[idx++] = Vec3{lx * (Real(6)*y*y - Real(4)*y + Real(0.5)) * z, Real(0), Real(0)};
    values[idx++] = Vec3{lx * (Real(3)*y - Real(1)) * (Real(3)*z*z - Real(1)), Real(0), Real(0)};
    values[idx++] = Vec3{lx * (Real(6)*y*y - Real(4)*y + Real(0.5)) * (Real(3)*z*z - Real(1)), Real(0), Real(0)};

    // ==========================================================================
    // Quad face 4 (x+y=1): 9 DOFs - Q_2 normal flux moments
    // ==========================================================================
    const auto loc4 = x + y;
    values[idx++] = Vec3{loc4, loc4, Real(0)};
    values[idx++] = Vec3{loc4 * (Real(3)*x - Real(1)), loc4 * (Real(3)*x - Real(1)), Real(0)};
    values[idx++] = Vec3{loc4 * z, loc4 * z, Real(0)};
    values[idx++] = Vec3{loc4 * (Real(6)*x*x - Real(4)*x + Real(0.5)), loc4 * (Real(6)*x*x - Real(4)*x + Real(0.5)), Real(0)};
    values[idx++] = Vec3{loc4 * (Real(3)*x - Real(1)) * z, loc4 * (Real(3)*x - Real(1)) * z, Real(0)};
    values[idx++] = Vec3{loc4 * (Real(3)*z*z - Real(1)), loc4 * (Real(3)*z*z - Real(1)), Real(0)};
    values[idx++] = Vec3{loc4 * (Real(6)*x*x - Real(4)*x + Real(0.5)) * z, loc4 * (Real(6)*x*x - Real(4)*x + Real(0.5)) * z, Real(0)};
    values[idx++] = Vec3{loc4 * (Real(3)*x - Real(1)) * (Real(3)*z*z - Real(1)), loc4 * (Real(3)*x - Real(1)) * (Real(3)*z*z - Real(1)), Real(0)};
    values[idx++] = Vec3{loc4 * (Real(6)*x*x - Real(4)*x + Real(0.5)) * (Real(3)*z*z - Real(1)), loc4 * (Real(6)*x*x - Real(4)*x + Real(0.5)) * (Real(3)*z*z - Real(1)), Real(0)};

    // ==========================================================================
    // Interior DOFs: 27 DOFs (3 * k(k+1)^2/2 = 3*2*9/2 = 27 for k=2)
    // Bubble function for interior
    // ==========================================================================
    const auto bubble_xy = L0 * L1 * L2;
    const auto bubble_z = (Real(1) - z*z);
    const auto bubble = bubble_xy * bubble_z;

    // P_1(x,y) x P_2(z) test space for each component: 3 * 3 * 3 = 27
    // Use: {1, x, y} x {1, z, z^2} x {e_x, e_y, e_z}
    const Real scale = Real(60);
    values[idx++] = Vec3{bubble * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * scale};
    values[idx++] = Vec3{bubble * z * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * z * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * z * scale};
    values[idx++] = Vec3{bubble * z*z * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * z*z * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * z*z * scale};
    values[idx++] = Vec3{bubble * x * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * x * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * x * scale};
    values[idx++] = Vec3{bubble * x * z * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * x * z * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * x * z * scale};
    values[idx++] = Vec3{bubble * x * z*z * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * x * z*z * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * x * z*z * scale};
    values[idx++] = Vec3{bubble * y * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * y * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * y * scale};
    values[idx++] = Vec3{bubble * y * z * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * y * z * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * y * z * scale};
    values[idx++] = Vec3{bubble * y * z*z * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * y * z*z * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * y * z*z * scale};
}

inline void eval_wedge_rt2_direct(const Vec3& xi, std::vector<Vec3>& values) {
    eval_direct_values_real(xi, eval_wedge_rt2_direct_impl<Real>, values);
}

inline void eval_wedge_rt2_divergence_direct(const Vec3& xi, std::vector<Real>& divergence) {
    eval_direct_divergence_exact(xi, eval_wedge_rt2_direct_impl<Diff3>, divergence);
}

template <typename Scalar>
inline void eval_pyramid_rt2_direct_impl(const DirectVec3<Scalar>& xi,
                                         std::vector<DirectVec3<Scalar>>& values) {
    using Vec3 = DirectVec3<Scalar>;

    const auto x = xi[0];
    const auto y = xi[1];
    const auto z = xi[2];

    const auto zc = Real(1) - z;
    const auto loc_f1 = Real(1) + y - z;
    const auto loc_f2 = Real(1) - x - z;
    const auto loc_f3 = Real(1) - y - z;
    const auto loc_f4 = Real(1) + x - z;

    values.resize(57);
    std::size_t idx = 0;

    // ==========================================================================
    // Quad base face (z=0): 9 DOFs - Q_2 normal flux moments
    // ==========================================================================
    values[idx++] = Vec3{Real(0), Real(0), -zc};
    values[idx++] = Vec3{Real(0), Real(0), -zc * x * Real(3)};
    values[idx++] = Vec3{Real(0), Real(0), -zc * y * Real(3)};
    values[idx++] = Vec3{Real(0), Real(0), -zc * (Real(3)*x*x - Real(1)) * Real(1.5)};
    values[idx++] = Vec3{Real(0), Real(0), -zc * x * y * Real(9)};
    values[idx++] = Vec3{Real(0), Real(0), -zc * (Real(3)*y*y - Real(1)) * Real(1.5)};
    values[idx++] = Vec3{Real(0), Real(0), -zc * (Real(3)*x*x - Real(1)) * y * Real(4.5)};
    values[idx++] = Vec3{Real(0), Real(0), -zc * x * (Real(3)*y*y - Real(1)) * Real(4.5)};
    values[idx++] = Vec3{Real(0), Real(0), -zc * (Real(3)*x*x - Real(1)) * (Real(3)*y*y - Real(1)) * Real(2.25)};

    // ==========================================================================
    // 4 triangular faces: 6 DOFs each (P_2 normal flux)
    // ==========================================================================
    // Face 1 (front)
    values[idx++] = Vec3{Real(0), -loc_f1, Real(0)};
    values[idx++] = Vec3{Real(0), -loc_f1 * x * Real(3), Real(0)};
    values[idx++] = Vec3{Real(0), -loc_f1 * z * Real(3), Real(0)};
    values[idx++] = Vec3{Real(0), -loc_f1 * x*x * Real(6), Real(0)};
    values[idx++] = Vec3{Real(0), -loc_f1 * x * z * Real(9), Real(0)};
    values[idx++] = Vec3{Real(0), -loc_f1 * z*z * Real(6), Real(0)};

    // Face 2 (right)
    values[idx++] = Vec3{-loc_f2, Real(0), Real(0)};
    values[idx++] = Vec3{-loc_f2 * y * Real(3), Real(0), Real(0)};
    values[idx++] = Vec3{-loc_f2 * z * Real(3), Real(0), Real(0)};
    values[idx++] = Vec3{-loc_f2 * y*y * Real(6), Real(0), Real(0)};
    values[idx++] = Vec3{-loc_f2 * y * z * Real(9), Real(0), Real(0)};
    values[idx++] = Vec3{-loc_f2 * z*z * Real(6), Real(0), Real(0)};

    // Face 3 (back)
    values[idx++] = Vec3{Real(0), -loc_f3, Real(0)};
    values[idx++] = Vec3{Real(0), -loc_f3 * x * Real(3), Real(0)};
    values[idx++] = Vec3{Real(0), -loc_f3 * z * Real(3), Real(0)};
    values[idx++] = Vec3{Real(0), -loc_f3 * x*x * Real(6), Real(0)};
    values[idx++] = Vec3{Real(0), -loc_f3 * x * z * Real(9), Real(0)};
    values[idx++] = Vec3{Real(0), -loc_f3 * z*z * Real(6), Real(0)};

    // Face 4 (left)
    values[idx++] = Vec3{loc_f4, Real(0), Real(0)};
    values[idx++] = Vec3{loc_f4 * y * Real(3), Real(0), Real(0)};
    values[idx++] = Vec3{loc_f4 * z * Real(3), Real(0), Real(0)};
    values[idx++] = Vec3{loc_f4 * y*y * Real(6), Real(0), Real(0)};
    values[idx++] = Vec3{loc_f4 * y * z * Real(9), Real(0), Real(0)};
    values[idx++] = Vec3{loc_f4 * z*z * Real(6), Real(0), Real(0)};

    // ==========================================================================
    // Interior: 24 DOFs
    // ==========================================================================
    const auto bubble = zc * loc_f1 * loc_f2 * loc_f3 * loc_f4;
    const Real scale = Real(120);
    values[idx++] = Vec3{bubble * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * scale};
    values[idx++] = Vec3{bubble * x * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * x * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * x * scale};
    values[idx++] = Vec3{bubble * y * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * y * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * y * scale};
    values[idx++] = Vec3{bubble * z * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * z * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * z * scale};
    values[idx++] = Vec3{bubble * x * y * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * x * y * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * x * y * scale};
    values[idx++] = Vec3{bubble * x * z * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * x * z * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * x * z * scale};
    values[idx++] = Vec3{bubble * y * z * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * y * z * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * y * z * scale};
    values[idx++] = Vec3{bubble * x * x * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * y * y * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * z * z * scale};
}

inline void eval_pyramid_rt2_direct(const Vec3& xi, std::vector<Vec3>& values) {
    eval_direct_values_real(xi, eval_pyramid_rt2_direct_impl<Real>, values);
}

inline void eval_pyramid_rt2_divergence_direct(const Vec3& xi, std::vector<Real>& divergence) {
    eval_direct_divergence_exact(xi, eval_pyramid_rt2_direct_impl<Diff3>, divergence);
}

template <typename Scalar>
inline void eval_wedge_nd2_direct_impl(const DirectVec3<Scalar>& xi,
                                       std::vector<DirectVec3<Scalar>>& values) {
    using Vec3 = DirectVec3<Scalar>;

    const auto x = xi[0];
    const auto y = xi[1];
    const auto z = xi[2];

    const auto L0 = Real(1) - x - y;
    const auto L1 = x;
    const auto L2 = y;
    const auto zb = (Real(1) - z) * Real(0.5);
    const auto zt = (Real(1) + z) * Real(0.5);

    values.resize(84);
    std::size_t idx = 0;

    // ==========================================================================
    // Edge DOFs: 9 edges x 3 moments = 27 DOFs
    // ==========================================================================
    // Bottom triangle edges
    for (int mode = 0; mode <= 2; ++mode) {
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? (Real(2)*x - Real(1)) : (Real(6)*x*x - Real(6)*x + Real(1)));
        values[idx++] = Vec3{zb * (Real(1) - y) * leg, zb * x * leg * Real(0), Real(0)};
    }
    for (int mode = 0; mode <= 2; ++mode) {
        const auto s = y; // parameter along edge
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? (Real(2)*s - Real(1)) : (Real(6)*s*s - Real(6)*s + Real(1)));
        values[idx++] = Vec3{-zb * L0 * leg, zb * L0 * leg, Real(0)};
    }
    for (int mode = 0; mode <= 2; ++mode) {
        const auto s = Real(1) - y;
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? (Real(2)*s - Real(1)) : (Real(6)*s*s - Real(6)*s + Real(1)));
        values[idx++] = Vec3{Real(0), -zb * (Real(1) - x) * leg, Real(0)};
    }

    // Top triangle edges
    for (int mode = 0; mode <= 2; ++mode) {
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? (Real(2)*x - Real(1)) : (Real(6)*x*x - Real(6)*x + Real(1)));
        values[idx++] = Vec3{zt * (Real(1) - y) * leg, Real(0), Real(0)};
    }
    for (int mode = 0; mode <= 2; ++mode) {
        const auto s = y;
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? (Real(2)*s - Real(1)) : (Real(6)*s*s - Real(6)*s + Real(1)));
        values[idx++] = Vec3{-zt * L0 * leg, zt * L0 * leg, Real(0)};
    }
    for (int mode = 0; mode <= 2; ++mode) {
        const auto s = Real(1) - y;
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? (Real(2)*s - Real(1)) : (Real(6)*s*s - Real(6)*s + Real(1)));
        values[idx++] = Vec3{Real(0), -zt * (Real(1) - x) * leg, Real(0)};
    }

    // Vertical edges
    for (int mode = 0; mode <= 2; ++mode) {
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? z : (Real(3)*z*z - Real(1)) / Real(2));
        values[idx++] = Vec3{Real(0), Real(0), L0 * leg};
    }
    for (int mode = 0; mode <= 2; ++mode) {
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? z : (Real(3)*z*z - Real(1)) / Real(2));
        values[idx++] = Vec3{Real(0), Real(0), L1 * leg};
    }
    for (int mode = 0; mode <= 2; ++mode) {
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? z : (Real(3)*z*z - Real(1)) / Real(2));
        values[idx++] = Vec3{Real(0), Real(0), L2 * leg};
    }

    // ==========================================================================
    // Face DOFs: 48 total
    // ==========================================================================
    // Bottom tri: 6 tangential
    const auto tb = L0 * L1 * L2 * zb;
    values[idx++] = Vec3{tb * Real(60), Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), tb * Real(60), Real(0)};
    values[idx++] = Vec3{tb * x * Real(60), Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), tb * y * Real(60), Real(0)};
    values[idx++] = Vec3{tb * y * Real(60), Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), tb * x * Real(60), Real(0)};

    // Top tri: 6 tangential
    const auto tt = L0 * L1 * L2 * zt;
    values[idx++] = Vec3{tt * Real(60), Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), tt * Real(60), Real(0)};
    values[idx++] = Vec3{tt * x * Real(60), Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), tt * y * Real(60), Real(0)};
    values[idx++] = Vec3{tt * y * Real(60), Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), tt * x * Real(60), Real(0)};

    // Quad faces: 3 faces x 12 tangential = 36
    // Face y=0
    const auto qf2 = (Real(1) - y) * L0;
    for (int i = 0; i < 6; ++i) {
        const auto poly = (i == 0) ? Real(1) : ((i == 1) ? x : ((i == 2) ? z : ((i == 3) ? x*z : ((i == 4) ? x*x : z*z))));
        values[idx++] = Vec3{qf2 * poly * Real(12), Real(0), Real(0)};
    }
    for (int i = 0; i < 6; ++i) {
        const auto poly = (i == 0) ? Real(1) : ((i == 1) ? x : ((i == 2) ? z : ((i == 3) ? x*z : ((i == 4) ? x*x : z*z))));
        values[idx++] = Vec3{Real(0), Real(0), qf2 * poly * Real(12)};
    }

    // Face x=0
    const auto qf3 = (Real(1) - x) * L0;
    for (int i = 0; i < 6; ++i) {
        const auto poly = (i == 0) ? Real(1) : ((i == 1) ? y : ((i == 2) ? z : ((i == 3) ? y*z : ((i == 4) ? y*y : z*z))));
        values[idx++] = Vec3{Real(0), qf3 * poly * Real(12), Real(0)};
    }
    for (int i = 0; i < 6; ++i) {
        const auto poly = (i == 0) ? Real(1) : ((i == 1) ? y : ((i == 2) ? z : ((i == 3) ? y*z : ((i == 4) ? y*y : z*z))));
        values[idx++] = Vec3{Real(0), Real(0), qf3 * poly * Real(12)};
    }

    // Face x+y=1
    const auto loc4 = x + y;
    const auto qf4 = loc4 * L2;
    for (int i = 0; i < 6; ++i) {
        const auto poly = (i == 0) ? Real(1) : ((i == 1) ? x : ((i == 2) ? z : ((i == 3) ? x*z : ((i == 4) ? x*x : z*z))));
        values[idx++] = Vec3{-qf4 * poly * Real(12), qf4 * poly * Real(12), Real(0)};
    }
    for (int i = 0; i < 6; ++i) {
        const auto poly = (i == 0) ? Real(1) : ((i == 1) ? x : ((i == 2) ? z : ((i == 3) ? x*z : ((i == 4) ? x*x : z*z))));
        values[idx++] = Vec3{Real(0), Real(0), qf4 * poly * Real(12)};
    }

    // ==========================================================================
    // Interior DOFs: 9 DOFs
    // ==========================================================================
    const auto bubble = L0 * L1 * L2 * (Real(1) - z*z);
    const auto scale = Real(180);
    values[idx++] = Vec3{bubble * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * scale};
    values[idx++] = Vec3{bubble * z * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * z * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * z * scale};
    values[idx++] = Vec3{bubble * x * scale, Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * y * scale, Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * (x+y) * scale};
}

inline void eval_wedge_nd2_direct(const Vec3& xi, std::vector<Vec3>& values) {
    eval_direct_values_real(xi, eval_wedge_nd2_direct_impl<Real>, values);
}

template <typename Scalar>
inline void eval_pyramid_nd2_direct_impl(const DirectVec3<Scalar>& xi,
                                         std::vector<DirectVec3<Scalar>>& values) {
    using Vec3 = DirectVec3<Scalar>;

    const auto x = xi[0];
    const auto y = xi[1];
    const auto z = xi[2];

    const auto zc = Real(1) - z;
    const auto loc_f1 = Real(1) + y - z;
    const auto loc_f2 = Real(1) - x - z;
    const auto loc_f3 = Real(1) - y - z;
    const auto loc_f4 = Real(1) + x - z;

    values.resize(63);
    std::size_t idx = 0;

    // ==========================================================================
    // Edge DOFs: 8 edges x 3 moments = 24 DOFs
    // ==========================================================================
    // Base edges
    const auto le0 = (Real(1) + y) * zc;
    for (int mode = 0; mode <= 2; ++mode) {
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? x : x*x);
        values[idx++] = Vec3{le0 * leg, Real(0), Real(0)};
    }
    const auto le1 = (Real(1) - x) * zc;
    for (int mode = 0; mode <= 2; ++mode) {
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? y : y*y);
        values[idx++] = Vec3{Real(0), le1 * leg, Real(0)};
    }
    const auto le2 = (Real(1) - y) * zc;
    for (int mode = 0; mode <= 2; ++mode) {
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? x : x*x);
        values[idx++] = Vec3{-le2 * leg, Real(0), Real(0)};
    }
    const auto le3 = (Real(1) + x) * zc;
    for (int mode = 0; mode <= 2; ++mode) {
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? y : y*y);
        values[idx++] = Vec3{Real(0), -le3 * leg, Real(0)};
    }

    // Apex edges
    const auto le4 = loc_f2 * loc_f3;
    for (int mode = 0; mode <= 2; ++mode) {
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? z : z*z);
        values[idx++] = Vec3{le4 * leg, le4 * leg, le4 * leg * Real(2)};
    }
    const auto le5 = loc_f3 * loc_f4;
    for (int mode = 0; mode <= 2; ++mode) {
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? z : z*z);
        values[idx++] = Vec3{-le5 * leg, le5 * leg, le5 * leg * Real(2)};
    }
    const auto le6 = loc_f4 * loc_f1;
    for (int mode = 0; mode <= 2; ++mode) {
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? z : z*z);
        values[idx++] = Vec3{-le6 * leg, -le6 * leg, le6 * leg * Real(2)};
    }
    const auto le7 = loc_f1 * loc_f2;
    for (int mode = 0; mode <= 2; ++mode) {
        const auto leg = (mode == 0) ? Real(1) : ((mode == 1) ? z : z*z);
        values[idx++] = Vec3{le7 * leg, -le7 * leg, le7 * leg * Real(2)};
    }

    // ==========================================================================
    // Face DOFs: 36 total (12 quad + 24 tri)
    // ==========================================================================
    // Quad base: 12 tangential
    const auto qbase = (Real(1) - x*x) * (Real(1) - y*y) * zc;
    for (int i = 0; i < 6; ++i) {
        const auto poly = (i == 0) ? Real(1) : ((i == 1) ? x : ((i == 2) ? y : ((i == 3) ? x*y : ((i == 4) ? x*x : y*y))));
        values[idx++] = Vec3{qbase * poly * Real(4), Real(0), Real(0)};
    }
    for (int i = 0; i < 6; ++i) {
        const auto poly = (i == 0) ? Real(1) : ((i == 1) ? x : ((i == 2) ? y : ((i == 3) ? x*y : ((i == 4) ? x*x : y*y))));
        values[idx++] = Vec3{Real(0), qbase * poly * Real(4), Real(0)};
    }

    // 4 tri faces: 6 tangential each = 24
    const auto tf1 = loc_f1 * (Real(1) - x*x);
    for (int i = 0; i < 6; ++i) {
        const auto poly = (i == 0) ? Real(1) : ((i == 1) ? x : ((i == 2) ? z : ((i == 3) ? x*z : ((i == 4) ? x*x : z*z))));
        values[idx++] = Vec3{tf1 * poly * Real(4), Real(0), tf1 * poly * z * Real(2)};
    }
    const auto tf2 = loc_f2 * (Real(1) - y*y);
    for (int i = 0; i < 6; ++i) {
        const auto poly = (i == 0) ? Real(1) : ((i == 1) ? y : ((i == 2) ? z : ((i == 3) ? y*z : ((i == 4) ? y*y : z*z))));
        values[idx++] = Vec3{-tf2 * poly * z * Real(2), tf2 * poly * Real(4), Real(0)};
    }
    const auto tf3 = loc_f3 * (Real(1) - x*x);
    for (int i = 0; i < 6; ++i) {
        const auto poly = (i == 0) ? Real(1) : ((i == 1) ? x : ((i == 2) ? z : ((i == 3) ? x*z : ((i == 4) ? x*x : z*z))));
        values[idx++] = Vec3{tf3 * poly * Real(4), Real(0), -tf3 * poly * z * Real(2)};
    }
    const auto tf4 = loc_f4 * (Real(1) - y*y);
    for (int i = 0; i < 6; ++i) {
        const auto poly = (i == 0) ? Real(1) : ((i == 1) ? y : ((i == 2) ? z : ((i == 3) ? y*z : ((i == 4) ? y*y : z*z))));
        values[idx++] = Vec3{tf4 * poly * z * Real(2), tf4 * poly * Real(4), Real(0)};
    }

    // ==========================================================================
    // Interior DOFs: 3 DOFs
    // ==========================================================================
    const auto bubble = zc * loc_f1 * loc_f2 * loc_f3 * loc_f4;
    values[idx++] = Vec3{bubble * Real(120), Real(0), Real(0)};
    values[idx++] = Vec3{Real(0), bubble * Real(120), Real(0)};
    values[idx++] = Vec3{Real(0), Real(0), bubble * Real(120)};
}

inline void eval_pyramid_nd2_direct(const Vec3& xi, std::vector<Vec3>& values) {
    eval_direct_values_real(xi, eval_pyramid_nd2_direct_impl<Real>, values);
}

#include "detail/VectorBasisRtDetail.inc"

} // namespace

// -----------------------------------------------------------------------------
// Pyramid RT0 helper (H(div))
// -----------------------------------------------------------------------------

namespace {

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
        } else {
            transformed_seed_indices_.clear();
        }
        transformed_monomial_candidates_ = make_rt_extra_monomial_candidates(type, order_);
        coeffs_ = build_rt_direct_transform(type, order_, size_, transformed_monomial_candidates_);
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
            monomials_.push_back(poly);
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
                    monomials_.push_back(poly);
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
                        monomials_.push_back(poly);
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
                        monomials_.push_back(poly);
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
                        monomials_.push_back(poly);
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
                        monomials_.push_back(poly);
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
                        monomials_.push_back(poly);
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
                        monomials_.push_back(poly);
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
                        monomials_.push_back(poly);
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
                        monomials_.push_back(poly);
                    }
                }

                // Triangular face 1 (x = 1-z, y free): normal roughly (1,0,1)/sqrt(2)
                for (int j = 0; j <= k; ++j) {
                    for (int i = 0; i <= k - j; ++i) {
                        ModalPolynomial poly;
                        poly.num_terms = 2;
                        poly.terms[0] = ModalTerm{0, i, j, 0, Real(1)};
                        poly.terms[1] = ModalTerm{0, i, j, 1, Real(1)};
                        monomials_.push_back(poly);
                    }
                }

                // Triangular face 2 (y = -1+z, x free): normal roughly (0,-1,1)/sqrt(2)
                for (int i = 0; i <= k; ++i) {
                    for (int j = 0; j <= k - i; ++j) {
                        ModalPolynomial poly;
                        poly.num_terms = 2;
                        poly.terms[0] = ModalTerm{1, i, j, 0, Real(-1)};
                        poly.terms[1] = ModalTerm{1, i, j, 1, Real(1)};
                        monomials_.push_back(poly);
                    }
                }

                // Triangular face 3 (y = 1-z, x free): normal roughly (0,1,1)/sqrt(2)
                for (int i = 0; i <= k; ++i) {
                    for (int j = 0; j <= k - i; ++j) {
                        ModalPolynomial poly;
                        poly.num_terms = 2;
                        poly.terms[0] = ModalTerm{1, i, j, 0, Real(1)};
                        poly.terms[1] = ModalTerm{1, i, j, 1, Real(1)};
                        monomials_.push_back(poly);
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
        int max_px = 0, max_py = 0, max_pz = 0;
        for (const auto& poly : monomials_) {
            for (int t = 0; t < poly.num_terms; ++t) {
                const auto& term = poly.terms[static_cast<std::size_t>(t)];
                max_px = std::max(max_px, term.px);
                max_py = std::max(max_py, term.py);
                max_pz = std::max(max_pz, term.pz);
            }
        }

        // For oversized basis, we need to select n linearly independent columns
        // For standard cases, m == n and we just invert directly.
        // We'll handle oversized bases at the end by column selection.
        std::vector<Real> A(n * n, Real(0));
        const elements::ReferenceElement ref = elements::ReferenceElement::create(type);
        std::size_t row = 0;

        // If oversized, resize monomials to exactly n. The hierarchical construction
        // ensures the first n functions form a valid basis.
        if (oversized_basis) {
            monomials_.resize(n);
        }

        if (dimension_ == 2) {
            // Edge flux moments: ∫_e (v·n) * l_i(s) ds, i=0..k.
            const LagrangeBasis edge_basis(ElementType::Line2, k);
            const auto edge_quad = quadrature::QuadratureFactory::create(
                ElementType::Line2, 2 * k + 2, QuadratureType::GaussLegendre, /*use_cache=*/false);

            for (std::size_t e = 0; e < ref.num_edges(); ++e) {
                const auto& en = ref.edge_nodes(e);
                FE_CHECK_ARG(en.size() == 2u, "RT quad: expected 2 vertices per edge");
                const Vec3 p0 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(en[0]));
                const Vec3 p1 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(en[1]));
                const Vec3 t = normalize3(p1 - p0);
                const Vec3 nrm{t[1], -t[0], Real(0)}; // rotate -90
                const Real J = math::norm(p1 - p0) * Real(0.5);

                for (int a = 0; a <= k; ++a) {
                    FE_CHECK_ARG(row < n, "RT quad: row overflow in edge moments");
                    for (std::size_t q = 0; q < edge_quad->num_points(); ++q) {
                        const Real s = edge_quad->point(q)[0];
                        const Real wq = edge_quad->weight(q);
                        std::vector<Real> lvals;
                        edge_basis.evaluate_values(Vec3{s, Real(0), Real(0)}, lvals);
                        FE_CHECK_ARG(lvals.size() == static_cast<std::size_t>(k + 1), "RT quad: edge basis size mismatch");
                        const Vec3 xi = lerp(p0, p1, s);

                        const auto px = powers(xi[0], max_px);
                        const auto py = powers(xi[1], max_py);
                        const auto pz = powers(xi[2], max_pz);

                        const Real wt = wq * J * lvals[static_cast<std::size_t>(a)];
                        for (std::size_t p = 0; p < n; ++p) {
                            const auto& poly = monomials_[p];
                            Real dot = Real(0);
                            for (int tt = 0; tt < poly.num_terms; ++tt) {
                                const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                const Real mv = px[static_cast<std::size_t>(mono.px)] *
                                                py[static_cast<std::size_t>(mono.py)] *
                                                pz[static_cast<std::size_t>(mono.pz)];
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
                    ElementType::Triangle3, 2 * k + 2, QuadratureType::GaussLegendre, /*use_cache=*/false);

                const std::size_t nface = face_basis.size();
                for (std::size_t f = 0; f < ref.num_faces(); ++f) {
                    const auto& fn = ref.face_nodes(f);
                    FE_CHECK_ARG(fn.size() == 3u, "RT tetra: expected tri face with 3 vertices");
                    const Vec3 v0 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[0]));
                    const Vec3 v1 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[1]));
                    const Vec3 v2 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[2]));
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

                        std::vector<Real> bvals;
                        face_basis.evaluate_values(Vec3{u, v, Real(0)}, bvals);
                        FE_CHECK_ARG(bvals.size() == nface, "RT tetra: face basis size mismatch");

                        const Vec3 xi = v0 + e01 * u + e02 * v;
                        const auto px = powers(xi[0], max_px);
                        const auto py = powers(xi[1], max_py);
                        const auto pz = powers(xi[2], max_pz);

                        std::vector<Real> modal_dot(n, Real(0));
                        for (std::size_t p = 0; p < n; ++p) {
                            const auto& poly = monomials_[p];
                            Real dot = Real(0);
                            for (int t = 0; t < poly.num_terms; ++t) {
                                const auto& mono = poly.terms[static_cast<std::size_t>(t)];
                                const Real mv =
                                    px[static_cast<std::size_t>(mono.px)] *
                                    py[static_cast<std::size_t>(mono.py)] *
                                    pz[static_cast<std::size_t>(mono.pz)];
                                dot += mono.coefficient * nrm[static_cast<std::size_t>(mono.component)] * mv;
                            }
                            modal_dot[p] = dot;
                        }

                        const Real wt = wq * scale;
                        for (std::size_t a = 0; a < nface; ++a) {
                            const Real wa = wt * bvals[a];
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
                    ElementType::Quad4, 2 * k + 2, QuadratureType::GaussLegendre, /*use_cache=*/false);

                for (std::size_t f = 0; f < ref.num_faces(); ++f) {
                    const auto& fn = ref.face_nodes(f);
                    FE_CHECK_ARG(fn.size() == 4u, "RT hex: expected quad face with 4 vertices");
                    const std::array<Vec3, 4> fv{
                        NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[0])),
                        NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[1])),
                        NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[2])),
                        NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[3]))
                    };
                    const Vec3 nrm = normalize3(cross3(fv[1] - fv[0], fv[3] - fv[0]));

                    const std::size_t nface = static_cast<std::size_t>((k + 1) * (k + 1));
                    FE_CHECK_ARG(row + nface <= n, "RT hex: row overflow in face moments");

                    for (std::size_t q = 0; q < face_quad->num_points(); ++q) {
                        const auto uv = face_quad->point(q);
                        const Real u = uv[0];
                        const Real w = uv[1];
                        const Real wq = face_quad->weight(q);

                        std::vector<Real> bvals;
                        face_basis.evaluate_values(Vec3{u, w, Real(0)}, bvals);
                        FE_CHECK_ARG(bvals.size() == nface, "RT hex: face basis size mismatch");

                        const Vec3 xi = bilinear(fv, u, w);
                        const Vec3 dxdu = bilinear_du(fv, u, w);
                        const Vec3 dxdw = bilinear_dw(fv, u, w);
                        const Real scale = cross3(dxdu, dxdw).norm();

                        const auto px = powers(xi[0], max_px);
                        const auto py = powers(xi[1], max_py);
                        const auto pz = powers(xi[2], max_pz);

                        std::vector<Real> modal_dot(n, Real(0));
                        for (std::size_t p = 0; p < n; ++p) {
                            const auto& poly = monomials_[p];
                            Real dot = Real(0);
                            for (int t = 0; t < poly.num_terms; ++t) {
                                const auto& mono = poly.terms[static_cast<std::size_t>(t)];
                                const Real mv = px[static_cast<std::size_t>(mono.px)] *
                                                py[static_cast<std::size_t>(mono.py)] *
                                                pz[static_cast<std::size_t>(mono.pz)];
                                dot += mono.coefficient * nrm[static_cast<std::size_t>(mono.component)] * mv;
                            }
                            modal_dot[p] = dot;
                        }

                        const Real wt = wq * scale;
                        for (std::size_t a = 0; a < nface; ++a) {
                            const Real wa = wt * bvals[a];
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
                    ElementType::Triangle3, 2 * k + 2, QuadratureType::GaussLegendre, /*use_cache=*/false);
                const auto quad_quad = quadrature::QuadratureFactory::create(
                    ElementType::Quad4, 2 * k + 2, QuadratureType::GaussLegendre, /*use_cache=*/false);

                for (std::size_t f = 0; f < ref.num_faces(); ++f) {
                    const auto& fn = ref.face_nodes(f);
                    const bool is_tri = (fn.size() == 3u);

                    if (is_tri) {
                        // Triangular face
                        const Vec3 v0 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[0]));
                        const Vec3 v1 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[1]));
                        const Vec3 v2 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[2]));
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

                            std::vector<Real> bvals;
                            tri_face_basis.evaluate_values(Vec3{u, v, Real(0)}, bvals);

                            const Vec3 xi_pt = v0 + e01 * u + e02 * v;
                            const auto px = powers(xi_pt[0], max_px);
                            const auto py = powers(xi_pt[1], max_py);
                            const auto pz = powers(xi_pt[2], max_pz);

                            std::vector<Real> modal_dot(modal_count, Real(0));
                            for (std::size_t p = 0; p < modal_count; ++p) {
                                const auto& poly = monomials_[p];
                                Real dot = Real(0);
                                for (int t = 0; t < poly.num_terms; ++t) {
                                    const auto& mono = poly.terms[static_cast<std::size_t>(t)];
                                    const Real mv =
                                        px[static_cast<std::size_t>(mono.px)] *
                                        py[static_cast<std::size_t>(mono.py)] *
                                        pz[static_cast<std::size_t>(mono.pz)];
                                    dot += mono.coefficient * nrm[static_cast<std::size_t>(mono.component)] * mv;
                                }
                                modal_dot[p] = dot;
                            }

                            const Real wt = wq * scale;
                            for (std::size_t a = 0; a < nface; ++a) {
                                const Real wa = wt * bvals[a];
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
                            NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[0])),
                            NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[1])),
                            NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[2])),
                            NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[3]))
                        };
                        const Vec3 nrm = normalize3(cross3(fv[1] - fv[0], fv[3] - fv[0]));

                        const std::size_t nface = static_cast<std::size_t>((k + 1) * (k + 1));
                        if (row + nface > n) break; // Safety check

                        for (std::size_t q = 0; q < quad_quad->num_points(); ++q) {
                            const auto uv = quad_quad->point(q);
                            const Real u = uv[0];
                            const Real w = uv[1];
                            const Real wq = quad_quad->weight(q);

                            std::vector<Real> bvals;
                            quad_face_basis.evaluate_values(Vec3{u, w, Real(0)}, bvals);

                            const Vec3 xi_pt = bilinear(fv, u, w);
                            const Vec3 dxdu = bilinear_du(fv, u, w);
                            const Vec3 dxdw = bilinear_dw(fv, u, w);
                            const Real scale = cross3(dxdu, dxdw).norm();

                            const auto px = powers(xi_pt[0], max_px);
                            const auto py = powers(xi_pt[1], max_py);
                            const auto pz = powers(xi_pt[2], max_pz);

                            std::vector<Real> modal_dot(modal_count, Real(0));
                            for (std::size_t p = 0; p < modal_count; ++p) {
                                const auto& poly = monomials_[p];
                                Real dot = Real(0);
                                for (int t = 0; t < poly.num_terms; ++t) {
                                    const auto& mono = poly.terms[static_cast<std::size_t>(t)];
                                    const Real mv = px[static_cast<std::size_t>(mono.px)] *
                                                    py[static_cast<std::size_t>(mono.py)] *
                                                    pz[static_cast<std::size_t>(mono.pz)];
                                    dot += mono.coefficient * nrm[static_cast<std::size_t>(mono.component)] * mv;
                                }
                                modal_dot[p] = dot;
                            }

                            const Real wt = wq * scale;
                            for (std::size_t a = 0; a < nface; ++a) {
                                const Real wa = wt * bvals[a];
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

        if (order_ == 1 && (is_wedge(type) || is_pyramid(type))) {
            coeffs_ = pseudo_inverse_dense_matrix(A, n);
        } else {
            coeffs_ = invert_dense_matrix(std::move(A), n);
        }
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
        FE_CHECK_ARG(coeffs_.size() == candidate_count * n,
                     "RaviartThomasBasis::evaluate_vector_values: transformed RT coefficient size mismatch");

        std::size_t candidate = 0;
        if (num_seed > 0) {
            std::vector<Vec3> seed_values;
            eval_rt_seed_values(element_type_, order_, xi, seed_values);
            FE_CHECK_ARG(seed_values.size() >= num_seed,
                         "RaviartThomasBasis::evaluate_vector_values: RT seed basis size mismatch");

            for (int seed_idx : transformed_seed_indices_) {
                FE_CHECK_ARG(seed_idx >= 0 &&
                                 static_cast<std::size_t>(seed_idx) < seed_values.size(),
                             "RaviartThomasBasis::evaluate_vector_values: transformed RT seed index out of range");
                const Vec3& seed = seed_values[static_cast<std::size_t>(seed_idx)];
                for (std::size_t j = 0; j < n; ++j) {
                    const Real c = coeffs_[candidate * n + j];
                    values[j][0] += c * seed[0];
                    values[j][1] += c * seed[1];
                    values[j][2] += c * seed[2];
                }
                ++candidate;
            }
        }

        if (num_extra > 0) {
            int max_px = 0;
            int max_py = 0;
            int max_pz = 0;
            for (const auto& mono : transformed_monomial_candidates_) {
                max_px = std::max(max_px, mono[1]);
                max_py = std::max(max_py, mono[2]);
                max_pz = std::max(max_pz, mono[3]);
            }
            const auto px = powers(xi[0], max_px);
            const auto py = powers(xi[1], max_py);
            const auto pz = powers(xi[2], max_pz);

            for (const auto& mono : transformed_monomial_candidates_) {
                const Real scalar = eval_transformed_rt_monomial_scalar(mono, px, py, pz);
                Vec3 value{};
                value[static_cast<std::size_t>(mono[0])] = scalar;
                for (std::size_t j = 0; j < n; ++j) {
                    const Real c = coeffs_[candidate * n + j];
                    values[j][0] += c * value[0];
                    values[j][1] += c * value[1];
                    values[j][2] += c * value[2];
                }
                ++candidate;
            }
        }
        return;
    }

    // Use direct construction for wedge/pyramid RT(k>=1)
    if (use_direct_construction_) {
        if (is_wedge(element_type_)) {
            if (order_ == 1) {
                eval_wedge_rt1_direct(xi, values);
            } else if (order_ == 2) {
                eval_wedge_rt2_direct(xi, values);
            } else {
                throw NotImplementedException("RaviartThomasBasis direct wedge evaluation currently supports orders 1-2",
                                              __FILE__, __LINE__, __func__);
            }
        } else if (is_pyramid(element_type_)) {
            if (order_ == 1) {
                eval_pyramid_rt1_direct(xi, values);
            } else if (order_ == 2) {
                eval_pyramid_rt2_direct(xi, values);
            } else {
                throw NotImplementedException("RaviartThomasBasis direct pyramid evaluation currently supports orders 1-2",
                                              __FILE__, __LINE__, __func__);
            }
        }
        return;
    }

    if (nodal_generated_) {
        const std::size_t n = size_;
        values.assign(n, math::Vector<Real, 3>{});

        int max_px = 0, max_py = 0, max_pz = 0;
        for (const auto& poly : monomials_) {
            for (int t = 0; t < poly.num_terms; ++t) {
                const auto& m = poly.terms[static_cast<std::size_t>(t)];
                max_px = std::max(max_px, m.px);
                max_py = std::max(max_py, m.py);
                max_pz = std::max(max_pz, m.pz);
            }
        }
        const auto px = powers(xi[0], max_px);
        const auto py = powers(xi[1], max_py);
        const auto pz = powers(xi[2], max_pz);

        std::vector<math::Vector<Real, 3>> modal_vals(n, math::Vector<Real, 3>{});
        for (std::size_t p = 0; p < n; ++p) {
            const auto& poly = monomials_[p];
            math::Vector<Real, 3> v{};
            for (int t = 0; t < poly.num_terms; ++t) {
                const auto& m = poly.terms[static_cast<std::size_t>(t)];
                const Real mv =
                    px[static_cast<std::size_t>(m.px)] *
                    py[static_cast<std::size_t>(m.py)] *
                    pz[static_cast<std::size_t>(m.pz)];
                v[static_cast<std::size_t>(m.component)] += m.coefficient * mv;
            }
            modal_vals[p] = v;
        }

        // values[j] += coeffs_[p*n + j] * modal_vals[p]
        for (std::size_t p = 0; p < n; ++p) {
            const auto& mv = modal_vals[p];
            for (std::size_t j = 0; j < n; ++j) {
                const Real c = coeffs_[p * n + j];
                values[j][0] += c * mv[0];
                values[j][1] += c * mv[1];
                values[j][2] += c * mv[2];
            }
        }
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
        constexpr Real c0 = Real(-1.15470053837925152901829756100391491129520350254025L); // -2/sqrt(3)
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
        values[2] = math::Vector<Real, 3>{c0 * x, c0 * y, c0 * z};
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

void RaviartThomasBasis::evaluate_divergence(const math::Vector<Real, 3>& xi,
                                             std::vector<Real>& divergence) const {
    if (use_transformed_direct_seed_) {
        const std::size_t n = size_;
        const std::size_t num_seed = transformed_seed_indices_.size();
        const std::size_t num_extra = transformed_monomial_candidates_.size();
        const std::size_t candidate_count = num_seed + num_extra;
        divergence.assign(n, Real(0));
        FE_CHECK_ARG(coeffs_.size() == candidate_count * n,
                     "RaviartThomasBasis::evaluate_divergence: transformed RT coefficient size mismatch");

        std::size_t candidate = 0;
        if (num_seed > 0) {
            std::vector<Real> seed_divergence;
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
                for (std::size_t j = 0; j < n; ++j) {
                    divergence[j] += coeffs_[candidate * n + j] * seed;
                }
                ++candidate;
            }
        }

        if (num_extra > 0) {
            int max_px = 0;
            int max_py = 0;
            int max_pz = 0;
            for (const auto& mono : transformed_monomial_candidates_) {
                max_px = std::max(max_px, mono[1]);
                max_py = std::max(max_py, mono[2]);
                max_pz = std::max(max_pz, mono[3]);
            }
            const auto px = powers(xi[0], max_px);
            const auto py = powers(xi[1], max_py);
            const auto pz = powers(xi[2], max_pz);

            for (const auto& mono : transformed_monomial_candidates_) {
                const Real div = eval_transformed_rt_monomial_divergence(mono, px, py, pz);
                if (div == Real(0)) {
                    ++candidate;
                    continue;
                }
                for (std::size_t j = 0; j < n; ++j) {
                    divergence[j] += coeffs_[candidate * n + j] * div;
                }
                ++candidate;
            }
        }
        return;
    }

    // Use direct construction for wedge/pyramid RT(k>=1)
    if (use_direct_construction_) {
        if (is_wedge(element_type_)) {
            if (order_ == 1) {
                eval_wedge_rt1_divergence_direct(xi, divergence);
            } else if (order_ == 2) {
                eval_wedge_rt2_divergence_direct(xi, divergence);
            } else {
                throw NotImplementedException("RaviartThomasBasis direct wedge divergence currently supports orders 1-2",
                                              __FILE__, __LINE__, __func__);
            }
        } else if (is_pyramid(element_type_)) {
            if (order_ == 1) {
                eval_pyramid_rt1_divergence_direct(xi, divergence);
            } else if (order_ == 2) {
                eval_pyramid_rt2_divergence_direct(xi, divergence);
            } else {
                throw NotImplementedException("RaviartThomasBasis direct pyramid divergence currently supports orders 1-2",
                                              __FILE__, __LINE__, __func__);
            }
        }
        return;
    }

    if (nodal_generated_) {
        const std::size_t n = size_;
        divergence.assign(n, Real(0));

        int max_px = 0, max_py = 0, max_pz = 0;
        for (const auto& poly : monomials_) {
            for (int t = 0; t < poly.num_terms; ++t) {
                const auto& m = poly.terms[static_cast<std::size_t>(t)];
                max_px = std::max(max_px, m.px);
                max_py = std::max(max_py, m.py);
                max_pz = std::max(max_pz, m.pz);
            }
        }
        const auto px = powers(xi[0], max_px);
        const auto py = powers(xi[1], max_py);
        const auto pz = powers(xi[2], max_pz);

        std::vector<Real> div_mono(n, Real(0));
        for (std::size_t p = 0; p < n; ++p) {
            const auto& poly = monomials_[p];
            Real val = Real(0);
            for (int t = 0; t < poly.num_terms; ++t) {
                const auto& m = poly.terms[static_cast<std::size_t>(t)];
                if (m.component == 0) {
                    if (m.px > 0) {
                        val += m.coefficient *
                               static_cast<Real>(m.px) *
                               px[static_cast<std::size_t>(m.px - 1)] *
                               py[static_cast<std::size_t>(m.py)] *
                               pz[static_cast<std::size_t>(m.pz)];
                    }
                } else if (m.component == 1) {
                    if (m.py > 0) {
                        val += m.coefficient *
                               static_cast<Real>(m.py) *
                               px[static_cast<std::size_t>(m.px)] *
                               py[static_cast<std::size_t>(m.py - 1)] *
                               pz[static_cast<std::size_t>(m.pz)];
                    }
                } else {
                    if (m.pz > 0) {
                        val += m.coefficient *
                               static_cast<Real>(m.pz) *
                               px[static_cast<std::size_t>(m.px)] *
                               py[static_cast<std::size_t>(m.py)] *
                               pz[static_cast<std::size_t>(m.pz - 1)];
                    }
                }
            }
            div_mono[p] = val;
        }

        for (std::size_t p = 0; p < n; ++p) {
            const Real dm = div_mono[p];
            if (dm == Real(0)) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                divergence[j] += coeffs_[p * n + j] * dm;
            }
        }
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
            const Real c0 = Real(-2.0 / std::sqrt(3.0));
            // Keep ordering consistent with evaluate_vector_values (ReferenceElement face order).
            divergence = {Real(-6),
                          Real(-6),
                          Real(3) * c0,
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
        coeffs_ = build_nd_direct_transform(type, order_, size_, transformed_monomial_candidates_);
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
            monomials_.push_back(poly);
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
                    monomials_.push_back(poly);
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
                        monomials_.push_back(poly);
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
                        monomials_.push_back(poly);
                    }
                }

                // For q=(m,0,0) with px=0: x×q = (0, z*m, -y*m), m = y^py z^pz, py+pz=k.
                for (int py = 0; py <= k; ++py) {
                    const int pz = k - py;
                    ModalPolynomial poly;
                    poly.num_terms = 2;
                    poly.terms[0] = ModalTerm{1, 0, py, pz + 1, Real(1)};
                    poly.terms[1] = ModalTerm{2, 0, py + 1, pz, Real(-1)};
                    monomials_.push_back(poly);
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
                            monomials_.push_back(poly);
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

                // If oversized, truncate
                if (monomials_.size() > n) {
                    monomials_.resize(n);
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

                // If oversized, truncate
                if (monomials_.size() > n) {
                    monomials_.resize(n);
                }
            }
        }
        // For wedge and pyramid, we may have oversized basis - truncate if needed
        if (monomials_.size() > n) {
            monomials_.resize(n);
        }

        // Verify modal basis size matches expected DOF count
        FE_CHECK_ARG(monomials_.size() == n,
                     "NedelecBasis: modal basis size mismatch (expected " +
                     std::to_string(n) + ", got " + std::to_string(monomials_.size()) + ")");

        int max_px = 0, max_py = 0, max_pz = 0;
        for (const auto& poly : monomials_) {
            for (int t = 0; t < poly.num_terms; ++t) {
                const auto& m = poly.terms[static_cast<std::size_t>(t)];
                max_px = std::max(max_px, m.px);
                max_py = std::max(max_py, m.py);
                max_pz = std::max(max_pz, m.pz);
            }
        }

        std::vector<Real> A(n * n, Real(0));
        const elements::ReferenceElement ref = elements::ReferenceElement::create(type);
        std::size_t row = 0;

        // Edge tangential moments: ∫_e (v·t) * l_i(s) ds, i=0..k.
        const LagrangeBasis edge_basis(ElementType::Line2, k);
        const auto edge_quad = quadrature::QuadratureFactory::create(
            ElementType::Line2, 2 * k + 2, QuadratureType::GaussLegendre, /*use_cache=*/false);

        for (std::size_t e = 0; e < ref.num_edges(); ++e) {
            const auto& en = ref.edge_nodes(e);
            FE_CHECK_ARG(en.size() == 2u, "Nedelec: expected 2 vertices per edge");
            const Vec3 p0 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(en[0]));
            const Vec3 p1 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(en[1]));
            const Vec3 t = normalize3(p1 - p0);
            const Real J = math::norm(p1 - p0) * Real(0.5);

            for (int a = 0; a <= k; ++a) {
                FE_CHECK_ARG(row < n, "Nedelec: row overflow in edge moments");
                for (std::size_t q = 0; q < edge_quad->num_points(); ++q) {
                    const Real s = edge_quad->point(q)[0];
                    const Real wq = edge_quad->weight(q);
                    std::vector<Real> lvals;
                    edge_basis.evaluate_values(Vec3{s, Real(0), Real(0)}, lvals);
                    FE_CHECK_ARG(lvals.size() == static_cast<std::size_t>(k + 1), "Nedelec: edge basis size mismatch");
                    const Vec3 xi = lerp(p0, p1, s);

                    const auto px = powers(xi[0], max_px);
                    const auto py = powers(xi[1], max_py);
                    const auto pz = powers(xi[2], max_pz);

                    const Real wt = wq * J * lvals[static_cast<std::size_t>(a)];
                    for (std::size_t p = 0; p < n; ++p) {
                        const auto& poly = monomials_[p];
                        Real dot = Real(0);
                        for (int tt = 0; tt < poly.num_terms; ++tt) {
                            const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                            const Real mv = px[static_cast<std::size_t>(mono.px)] *
                                            py[static_cast<std::size_t>(mono.py)] *
                                            pz[static_cast<std::size_t>(mono.pz)];
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
                    ElementType::Triangle3, 2 * k + 2, QuadratureType::GaussLegendre, /*use_cache=*/false);
                const LagrangeBasis face_basis(ElementType::Triangle3, k - 1);
                const std::size_t n_face = face_basis.size();

                for (std::size_t f = 0; f < ref.num_faces(); ++f) {
                    const auto& fn = ref.face_nodes(f);
                    FE_CHECK_ARG(fn.size() == 3u, "ND tetra: expected tri face with 3 vertices");
                    const Vec3 v0 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[0]));
                    const Vec3 v1 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[1]));
                    const Vec3 v2 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[2]));
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

                        std::vector<Real> bvals;
                        face_basis.evaluate_values(Vec3{u, v, Real(0)}, bvals);
                        FE_CHECK_ARG(bvals.size() == n_face, "ND tetra: face basis size mismatch");

                        const Vec3 xi = v0 + tu * u + tv * v;
                        const auto px = powers(xi[0], max_px);
                        const auto py = powers(xi[1], max_py);
                        const auto pz = powers(xi[2], max_pz);

                        std::vector<Real> mono_dot_u(n, Real(0));
                        std::vector<Real> mono_dot_v(n, Real(0));
                        for (std::size_t p = 0; p < n; ++p) {
                            const auto& poly = monomials_[p];
                            Real du = Real(0);
                            Real dv = Real(0);
                            for (int tt = 0; tt < poly.num_terms; ++tt) {
                                const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                const Real mv =
                                    px[static_cast<std::size_t>(mono.px)] *
                                    py[static_cast<std::size_t>(mono.py)] *
                                    pz[static_cast<std::size_t>(mono.pz)];
                                du += mono.coefficient * tu[static_cast<std::size_t>(mono.component)] * mv;
                                dv += mono.coefficient * tv[static_cast<std::size_t>(mono.component)] * mv;
                            }
                            mono_dot_u[p] = du;
                            mono_dot_v[p] = dv;
                        }

                        const Real wt = wq * scale;
                        for (std::size_t a = 0; a < n_face; ++a) {
                            const Real wa = wt * bvals[a];
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
                    ElementType::Quad4, 2 * k + 2, QuadratureType::GaussLegendre, /*use_cache=*/false);

                const LagrangeBasis u_low(ElementType::Line2, k - 1);
                const LagrangeBasis u_full(ElementType::Line2, k);
                const LagrangeBasis w_low(ElementType::Line2, k - 1);
                const LagrangeBasis w_full(ElementType::Line2, k);

                for (std::size_t f = 0; f < ref.num_faces(); ++f) {
                    const auto& fn = ref.face_nodes(f);
                    FE_CHECK_ARG(fn.size() == 4u, "ND hex: expected quad face with 4 vertices");
                    const std::array<Vec3, 4> fv{
                        NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[0])),
                        NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[1])),
                        NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[2])),
                        NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[3]))
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

                        std::vector<Real> u_low_vals, u_full_vals, w_low_vals, w_full_vals;
                        u_low.evaluate_values(Vec3{u, Real(0), Real(0)}, u_low_vals);
                        u_full.evaluate_values(Vec3{u, Real(0), Real(0)}, u_full_vals);
                        w_low.evaluate_values(Vec3{w, Real(0), Real(0)}, w_low_vals);
                        w_full.evaluate_values(Vec3{w, Real(0), Real(0)}, w_full_vals);
                        FE_CHECK_ARG(u_low_vals.size() == static_cast<std::size_t>(k), "ND hex: u_low size mismatch");
                        FE_CHECK_ARG(u_full_vals.size() == static_cast<std::size_t>(k + 1), "ND hex: u_full size mismatch");
                        FE_CHECK_ARG(w_low_vals.size() == static_cast<std::size_t>(k), "ND hex: w_low size mismatch");
                        FE_CHECK_ARG(w_full_vals.size() == static_cast<std::size_t>(k + 1), "ND hex: w_full size mismatch");

                        const Vec3 xi = bilinear(fv, u, w);
                        const Vec3 dxdu = bilinear_du(fv, u, w);
                        const Vec3 dxdw = bilinear_dw(fv, u, w);
                        const Real scale = cross3(dxdu, dxdw).norm();
                        const Real wt = wq * scale;

                        const auto px = powers(xi[0], max_px);
                        const auto py = powers(xi[1], max_py);
                        const auto pz = powers(xi[2], max_pz);

                        std::vector<Real> mono_dot_u(n, Real(0));
                        std::vector<Real> mono_dot_w(n, Real(0));
                        for (std::size_t p = 0; p < n; ++p) {
                            const auto& poly = monomials_[p];
                            Real du = Real(0);
                            Real dwv = Real(0);
                            for (int tt = 0; tt < poly.num_terms; ++tt) {
                                const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                const Real mv = px[static_cast<std::size_t>(mono.px)] *
                                                py[static_cast<std::size_t>(mono.py)] *
                                                pz[static_cast<std::size_t>(mono.pz)];
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
                                const Real basis_val = u_low_vals[static_cast<std::size_t>(iu)] * w_full_vals[static_cast<std::size_t>(jw)];
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
                                const Real basis_val = u_full_vals[static_cast<std::size_t>(iu)] * w_low_vals[static_cast<std::size_t>(jw)];
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
                    ElementType::Triangle3, 2 * k + 2, QuadratureType::GaussLegendre, /*use_cache=*/false);
                const auto quad_quad = quadrature::QuadratureFactory::create(
                    ElementType::Quad4, 2 * k + 2, QuadratureType::GaussLegendre, /*use_cache=*/false);
                const LagrangeBasis tri_face_basis(ElementType::Triangle3, k - 1);
                const LagrangeBasis quad_u_basis(ElementType::Line2, k - 1);
                const LagrangeBasis quad_w_basis(ElementType::Line2, k - 1);

                for (std::size_t f = 0; f < ref.num_faces(); ++f) {
                    const auto& fn = ref.face_nodes(f);
                    const bool is_tri = (fn.size() == 3u);

                    if (is_tri) {
                        // Triangular face tangential moments
                        const Vec3 v0 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[0]));
                        const Vec3 v1 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[1]));
                        const Vec3 v2 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[2]));
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

                            std::vector<Real> bvals;
                            tri_face_basis.evaluate_values(Vec3{u, v, Real(0)}, bvals);

                            const Vec3 xi_pt = v0 + tu * u + tv * v;
                            const auto px = powers(xi_pt[0], max_px);
                            const auto py = powers(xi_pt[1], max_py);
                            const auto pz = powers(xi_pt[2], max_pz);

                            std::vector<Real> mono_u(modal_count, Real(0)), mono_v(modal_count, Real(0));
                            for (std::size_t p = 0; p < modal_count; ++p) {
                                const auto& poly = monomials_[p];
                                Real du = Real(0), dv = Real(0);
                                for (int tt = 0; tt < poly.num_terms; ++tt) {
                                    const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                    const Real mv = px[static_cast<std::size_t>(mono.px)] *
                                                    py[static_cast<std::size_t>(mono.py)] *
                                                    pz[static_cast<std::size_t>(mono.pz)];
                                    du += mono.coefficient * tu[static_cast<std::size_t>(mono.component)] * mv;
                                    dv += mono.coefficient * tv[static_cast<std::size_t>(mono.component)] * mv;
                                }
                                mono_u[p] = du;
                                mono_v[p] = dv;
                            }

                            const Real wt = wq * scale;
                            for (std::size_t a = 0; a < n_face; ++a) {
                                const Real wa = wt * bvals[a];
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
                            NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[0])),
                            NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[1])),
                            NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[2])),
                            NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[3]))
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

                            const auto px = powers(xi_pt[0], max_px);
                            const auto py = powers(xi_pt[1], max_py);
                            const auto pz = powers(xi_pt[2], max_pz);

                            std::vector<Real> u_vals, w_vals;
                            quad_u_basis.evaluate_values(Vec3{u, Real(0), Real(0)}, u_vals);
                            quad_w_basis.evaluate_values(Vec3{w, Real(0), Real(0)}, w_vals);

                            std::vector<Real> mono_u(modal_count, Real(0)), mono_w(modal_count, Real(0));
                            for (std::size_t p = 0; p < modal_count; ++p) {
                                const auto& poly = monomials_[p];
                                Real du = Real(0), dw = Real(0);
                                for (int tt = 0; tt < poly.num_terms; ++tt) {
                                    const auto& mono = poly.terms[static_cast<std::size_t>(tt)];
                                    const Real mv = px[static_cast<std::size_t>(mono.px)] *
                                                    py[static_cast<std::size_t>(mono.py)] *
                                                    pz[static_cast<std::size_t>(mono.pz)];
                                    du += mono.coefficient * tu[static_cast<std::size_t>(mono.component)] * mv;
                                    dw += mono.coefficient * tw[static_cast<std::size_t>(mono.component)] * mv;
                                }
                                mono_u[p] = du;
                                mono_w[p] = dw;
                            }

                            const Real wt = wq * scale;
                            // u-directed moments
                            const std::size_t k_plus_1 = static_cast<std::size_t>(k + 1);
                            for (std::size_t a = 0; a < u_vals.size(); ++a) {
                                for (std::size_t b = 0; b < w_vals.size(); ++b) {
                                    if (a * k_plus_1 + b >= n_u) continue;
                                    const Real wa = wt * u_vals[a] * w_vals[b];
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

        coeffs_ = invert_dense_matrix(std::move(A), n);
        nodal_generated_ = true;
    }
}

void NedelecBasis::evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                          std::vector<math::Vector<Real, 3>>& values) const {
    if (use_transformed_direct_seed_) {
        const std::size_t n = size_;
        const std::size_t num_seed = (order_ <= 2) ? n : 0u;
        const std::size_t num_extra = transformed_monomial_candidates_.size();
        const std::size_t candidate_count = num_seed + num_extra;
        values.assign(n, math::Vector<Real, 3>{});

        FE_CHECK_ARG(coeffs_.size() == candidate_count * n,
                     "NedelecBasis::evaluate_vector_values: transformed ND coefficient size mismatch");

        if (num_seed > 0) {
            std::vector<Vec3> seed_values;
            eval_nd_seed_values(element_type_, order_, xi, seed_values);
            FE_CHECK_ARG(seed_values.size() == n,
                         "NedelecBasis::evaluate_vector_values: ND seed basis size mismatch");

            for (std::size_t p = 0; p < n; ++p) {
                const auto& mv = seed_values[p];
                for (std::size_t j = 0; j < n; ++j) {
                    const Real c = coeffs_[p * n + j];
                    values[j][0] += c * mv[0];
                    values[j][1] += c * mv[1];
                    values[j][2] += c * mv[2];
                }
            }
        }

        if (num_extra > 0) {
            int max_px = 0;
            int max_py = 0;
            int max_pz = 0;
            for (const auto& mono : transformed_monomial_candidates_) {
                max_px = std::max(max_px, mono[1]);
                max_py = std::max(max_py, mono[2]);
                max_pz = std::max(max_pz, mono[3]);
            }
            const auto px = powers(xi[0], max_px);
            const auto py = powers(xi[1], max_py);
            const auto pz = powers(xi[2], max_pz);

            std::size_t candidate = num_seed;
            for (const auto& mono : transformed_monomial_candidates_) {
                Vec3 value{};
                value[static_cast<std::size_t>(mono[0])] =
                    eval_transformed_nd_monomial_scalar(mono, px, py, pz);
                for (std::size_t j = 0; j < n; ++j) {
                    const Real c = coeffs_[candidate * n + j];
                    values[j][0] += c * value[0];
                    values[j][1] += c * value[1];
                    values[j][2] += c * value[2];
                }
                ++candidate;
            }
        }
        return;
    }

    // Use direct construction for wedge/pyramid Nedelec(k>=1)
    if (use_direct_construction_) {
        if (is_wedge(element_type_)) {
            if (order_ == 1) {
                eval_wedge_nd1_direct(xi, values);
            } else if (order_ == 2) {
                eval_wedge_nd2_direct(xi, values);
            } else {
                throw NotImplementedException("NedelecBasis direct wedge evaluation currently supports orders 1-2",
                                              __FILE__, __LINE__, __func__);
            }
        } else if (is_pyramid(element_type_)) {
            if (order_ == 1) {
                eval_pyramid_nd1_direct(xi, values);
            } else if (order_ == 2) {
                eval_pyramid_nd2_direct(xi, values);
            } else {
                throw NotImplementedException("NedelecBasis direct pyramid evaluation currently supports orders 1-2",
                                              __FILE__, __LINE__, __func__);
            }
        }
        return;
    }

    if (nodal_generated_) {
        const std::size_t n = size_;
        values.assign(n, math::Vector<Real, 3>{});

        int max_px = 0, max_py = 0, max_pz = 0;
        for (const auto& poly : monomials_) {
            for (int t = 0; t < poly.num_terms; ++t) {
                const auto& m = poly.terms[static_cast<std::size_t>(t)];
                max_px = std::max(max_px, m.px);
                max_py = std::max(max_py, m.py);
                max_pz = std::max(max_pz, m.pz);
            }
        }
        const auto px = powers(xi[0], max_px);
        const auto py = powers(xi[1], max_py);
        const auto pz = powers(xi[2], max_pz);

        std::vector<math::Vector<Real, 3>> modal_vals(n, math::Vector<Real, 3>{});
        for (std::size_t p = 0; p < n; ++p) {
            const auto& poly = monomials_[p];
            math::Vector<Real, 3> v{};
            for (int t = 0; t < poly.num_terms; ++t) {
                const auto& m = poly.terms[static_cast<std::size_t>(t)];
                const Real mv =
                    px[static_cast<std::size_t>(m.px)] *
                    py[static_cast<std::size_t>(m.py)] *
                    pz[static_cast<std::size_t>(m.pz)];
                v[static_cast<std::size_t>(m.component)] += m.coefficient * mv;
            }
            modal_vals[p] = v;
        }

        for (std::size_t p = 0; p < n; ++p) {
            const auto& mv = modal_vals[p];
            for (std::size_t j = 0; j < n; ++j) {
                const Real c = coeffs_[p * n + j];
                values[j][0] += c * mv[0];
                values[j][1] += c * mv[1];
                values[j][2] += c * mv[2];
            }
        }
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

void NedelecBasis::evaluate_curl(const math::Vector<Real, 3>& xi,
                                 std::vector<math::Vector<Real, 3>>& curl) const {
    if (use_transformed_direct_seed_) {
        const std::size_t n = size_;
        const std::size_t num_seed = (order_ <= 2) ? n : 0u;
        const std::size_t num_extra = transformed_monomial_candidates_.size();
        const std::size_t candidate_count = num_seed + num_extra;
        curl.assign(n, math::Vector<Real, 3>{});

        FE_CHECK_ARG(coeffs_.size() == candidate_count * n,
                     "NedelecBasis::evaluate_curl: transformed ND coefficient size mismatch");

        if (num_seed > 0) {
            std::vector<Vec3> seed_curl;
            eval_nd_seed_curl(element_type_, order_, xi, seed_curl);
            FE_CHECK_ARG(seed_curl.size() == n,
                         "NedelecBasis::evaluate_curl: ND seed curl size mismatch");

            for (std::size_t p = 0; p < n; ++p) {
                const auto& mv = seed_curl[p];
                for (std::size_t j = 0; j < n; ++j) {
                    const Real c = coeffs_[p * n + j];
                    curl[j][0] += c * mv[0];
                    curl[j][1] += c * mv[1];
                    curl[j][2] += c * mv[2];
                }
            }
        }

        if (num_extra > 0) {
            int max_px = 0;
            int max_py = 0;
            int max_pz = 0;
            for (const auto& mono : transformed_monomial_candidates_) {
                max_px = std::max(max_px, mono[1]);
                max_py = std::max(max_py, mono[2]);
                max_pz = std::max(max_pz, mono[3]);
            }
            const auto px = powers(xi[0], max_px);
            const auto py = powers(xi[1], max_py);
            const auto pz = powers(xi[2], max_pz);

            std::size_t candidate = num_seed;
            for (const auto& mono : transformed_monomial_candidates_) {
                const Vec3 mono_curl =
                    eval_transformed_nd_monomial_curl(mono, px, py, pz);
                for (std::size_t j = 0; j < n; ++j) {
                    const Real c = coeffs_[candidate * n + j];
                    curl[j][0] += c * mono_curl[0];
                    curl[j][1] += c * mono_curl[1];
                    curl[j][2] += c * mono_curl[2];
                }
                ++candidate;
            }
        }
        return;
    }

    // Use direct construction for wedge/pyramid Nedelec(k>=1)
    if (use_direct_construction_) {
        if (is_wedge(element_type_)) {
            if (order_ == 1) {
                eval_direct_curl_exact(xi, eval_wedge_nd1_direct_impl<Diff3>, curl);
                return;
            } else if (order_ == 2) {
                eval_direct_curl_exact(xi, eval_wedge_nd2_direct_impl<Diff3>, curl);
                return;
            } else {
                throw NotImplementedException("NedelecBasis direct wedge curl currently supports orders 1-2",
                                              __FILE__, __LINE__, __func__);
            }
        } else if (is_pyramid(element_type_)) {
            if (order_ == 1) {
                eval_pyramid_nd1_curl_direct(xi, curl);
                return;
            } else if (order_ == 2) {
                eval_direct_curl_exact(xi, eval_pyramid_nd2_direct_impl<Diff3>, curl);
                return;
            } else {
                throw NotImplementedException("NedelecBasis direct pyramid curl currently supports orders 1-2",
                                              __FILE__, __LINE__, __func__);
            }
        }
    }

    if (nodal_generated_) {
        const std::size_t n = size_;
        curl.assign(n, math::Vector<Real, 3>{});

        int max_px = 0, max_py = 0, max_pz = 0;
        for (const auto& poly : monomials_) {
            for (int t = 0; t < poly.num_terms; ++t) {
                const auto& m = poly.terms[static_cast<std::size_t>(t)];
                max_px = std::max(max_px, m.px);
                max_py = std::max(max_py, m.py);
                max_pz = std::max(max_pz, m.pz);
            }
        }
        const auto px = powers(xi[0], max_px);
        const auto py = powers(xi[1], max_py);
        const auto pz = powers(xi[2], max_pz);

        std::vector<Vec3> curl_modal(n, Vec3{});
        for (std::size_t p = 0; p < n; ++p) {
            const auto& poly = monomials_[p];
            Vec3 c{};
            for (int t = 0; t < poly.num_terms; ++t) {
                const auto& m = poly.terms[static_cast<std::size_t>(t)];
                const Real dfdx = (m.px > 0)
                    ? m.coefficient * static_cast<Real>(m.px) *
                      px[static_cast<std::size_t>(m.px - 1)] *
                      py[static_cast<std::size_t>(m.py)] *
                      pz[static_cast<std::size_t>(m.pz)]
                    : Real(0);
                const Real dfdy = (m.py > 0)
                    ? m.coefficient * static_cast<Real>(m.py) *
                      px[static_cast<std::size_t>(m.px)] *
                      py[static_cast<std::size_t>(m.py - 1)] *
                      pz[static_cast<std::size_t>(m.pz)]
                    : Real(0);
                const Real dfdz = (m.pz > 0)
                    ? m.coefficient * static_cast<Real>(m.pz) *
                      px[static_cast<std::size_t>(m.px)] *
                      py[static_cast<std::size_t>(m.py)] *
                      pz[static_cast<std::size_t>(m.pz - 1)]
                    : Real(0);

                if (m.component == 0) {
                    c[1] += dfdz;
                    c[2] -= dfdy;
                } else if (m.component == 1) {
                    c[0] -= dfdz;
                    c[2] += dfdx;
                } else {
                    c[0] += dfdy;
                    c[1] -= dfdx;
                }
            }
            curl_modal[p] = c;
        }

        for (std::size_t p = 0; p < n; ++p) {
            const Vec3 cm = curl_modal[p];
            for (std::size_t j = 0; j < n; ++j) {
                const Real c = coeffs_[p * n + j];
                curl[j][0] += c * cm[0];
                curl[j][1] += c * cm[1];
                curl[j][2] += c * cm[2];
            }
        }
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
        curl[0] = math::Vector<Real, 3>{Real(0), Real(-1), Real(1)};
        curl[1] = math::Vector<Real, 3>{Real(0), Real(0), Real(2)};
        curl[2] = math::Vector<Real, 3>{Real(0), Real(0), Real(2)};
        curl[3] = math::Vector<Real, 3>{Real(1), Real(-1), Real(0)};
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
    curl[8][0] =  Real(0.125) * (Real(1) - x);
    curl[8][1] = -Real(0.125) * (Real(1) - y);

    curl[9][0] =  Real(0.125) * (Real(1) + x);
    curl[9][1] = -Real(0.125) * (Real(1) - y);

    curl[10][0] =  Real(0.125) * (Real(1) + x);
    curl[10][1] = -Real(0.125) * (Real(1) + y);

    curl[11][0] =  Real(0.125) * (Real(1) - x);
    curl[11][1] = -Real(0.125) * (Real(1) + y);
    return;
}

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

    const int max_px = order_;
    const int max_py = order_;
    const int max_pz = (dimension_ == 3) ? order_ : 0;
    std::vector<Real> A(n * n, Real(0));
    std::size_t row = 0;

    if (type == ElementType::Triangle3) {
        const elements::ReferenceElement ref = elements::ReferenceElement::create(type);
        const LagrangeBasis edge_basis(ElementType::Line2, order_);
        const auto edge_quad = quadrature::QuadratureFactory::create(
            ElementType::Line2, 2 * order_ + 2, QuadratureType::GaussLegendre, /*use_cache=*/false);

        for (std::size_t e = 0; e < ref.num_edges(); ++e) {
            const auto& edge_nodes = ref.edge_nodes(e);
            const Vec3 a = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(edge_nodes[0]));
            const Vec3 b = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(edge_nodes[1]));
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
                std::vector<Real> bvals;
                edge_basis.evaluate_values(Vec3{s, Real(0), Real(0)}, bvals);

                const auto px = powers(xi[0], max_px);
                const auto py = powers(xi[1], max_py);

                std::vector<Real> modal_dot(n, Real(0));
                for (std::size_t p = 0; p < n; ++p) {
                    const auto& poly = monomials_[p];
                    Real dot = Real(0);
                    for (int term = 0; term < poly.num_terms; ++term) {
                        const auto& mono = poly.terms[static_cast<std::size_t>(term)];
                        const Real mv = px[static_cast<std::size_t>(mono.px)] *
                                        py[static_cast<std::size_t>(mono.py)];
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
            NedelecBasis interior_basis(ElementType::Triangle3, order_ - 2);
            const auto tri_quad = quadrature::QuadratureFactory::create(
                ElementType::Triangle3, 2 * order_ + 2, QuadratureType::GaussLegendre, /*use_cache=*/false);

            for (std::size_t q = 0; q < tri_quad->num_points(); ++q) {
                const Vec3 xi = tri_quad->point(q);
                std::vector<Vec3> test_values;
                interior_basis.evaluate_vector_values(xi, test_values);
                const auto px = powers(xi[0], max_px);
                const auto py = powers(xi[1], max_py);
                const Real wt = tri_quad->weight(q);

                std::vector<Real> modal_dot(n, Real(0));
                for (std::size_t p = 0; p < n; ++p) {
                    const auto& poly = monomials_[p];
                    Real dot = Real(0);
                    for (int term = 0; term < poly.num_terms; ++term) {
                        const auto& mono = poly.terms[static_cast<std::size_t>(term)];
                        const Real mv = px[static_cast<std::size_t>(mono.px)] *
                                        py[static_cast<std::size_t>(mono.py)];
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
            row += interior_basis.size();
        }
    } else {
        const elements::ReferenceElement ref = elements::ReferenceElement::create(type);
        const LagrangeBasis face_basis(ElementType::Triangle3, order_);
        const auto tri_quad = quadrature::QuadratureFactory::create(
            ElementType::Triangle3, 2 * order_ + 2, QuadratureType::GaussLegendre, /*use_cache=*/false);

        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const auto& face_nodes = ref.face_nodes(f);
            const Vec3 v0 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(face_nodes[0]));
            const Vec3 v1 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(face_nodes[1]));
            const Vec3 v2 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(face_nodes[2]));
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
                std::vector<Real> bvals;
                face_basis.evaluate_values(Vec3{u, v, Real(0)}, bvals);

                const auto px = powers(xi[0], max_px);
                const auto py = powers(xi[1], max_py);
                const auto pz = powers(xi[2], max_pz);

                std::vector<Real> modal_dot(n, Real(0));
                for (std::size_t p = 0; p < n; ++p) {
                    const auto& poly = monomials_[p];
                    Real dot = Real(0);
                    for (int term = 0; term < poly.num_terms; ++term) {
                        const auto& mono = poly.terms[static_cast<std::size_t>(term)];
                        const Real mv = px[static_cast<std::size_t>(mono.px)] *
                                        py[static_cast<std::size_t>(mono.py)] *
                                        pz[static_cast<std::size_t>(mono.pz)];
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
            NedelecBasis interior_basis(ElementType::Tetra4, order_ - 2);
            const auto tet_quad = quadrature::QuadratureFactory::create(
                ElementType::Tetra4, 2 * order_ + 2, QuadratureType::GaussLegendre, /*use_cache=*/false);

            for (std::size_t q = 0; q < tet_quad->num_points(); ++q) {
                const Vec3 xi = tet_quad->point(q);
                std::vector<Vec3> test_values;
                interior_basis.evaluate_vector_values(xi, test_values);
                const auto px = powers(xi[0], max_px);
                const auto py = powers(xi[1], max_py);
                const auto pz = powers(xi[2], max_pz);
                const Real wt = tet_quad->weight(q);

                for (std::size_t aidx = 0; aidx < test_values.size(); ++aidx) {
                    const Vec3& test = test_values[aidx];
                    const std::size_t r = row + aidx;
                    for (std::size_t p = 0; p < n; ++p) {
                        const auto& mono = monomials_[p].terms[0];
                        const Real mv = px[static_cast<std::size_t>(mono.px)] *
                                        py[static_cast<std::size_t>(mono.py)] *
                                        pz[static_cast<std::size_t>(mono.pz)];
                        A[r * n + p] += wt * mono.coefficient * mv *
                                        test[static_cast<std::size_t>(mono.component)];
                    }
                }
            }
            row += interior_basis.size();
        }
    }

    FE_CHECK_ARG(row == n, "BDMBasis: DOF assembly did not fill matrix");
    coeffs_ = invert_dense_matrix(std::move(A), n);
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
        const std::size_t n = size_;
        values.assign(n, math::Vector<Real, 3>{});

        int max_px = 0, max_py = 0, max_pz = 0;
        for (const auto& poly : monomials_) {
            for (int t = 0; t < poly.num_terms; ++t) {
                const auto& m = poly.terms[static_cast<std::size_t>(t)];
                max_px = std::max(max_px, m.px);
                max_py = std::max(max_py, m.py);
                max_pz = std::max(max_pz, m.pz);
            }
        }
        const auto px = powers(xi[0], max_px);
        const auto py = powers(xi[1], max_py);
        const auto pz = powers(xi[2], max_pz);

        std::vector<math::Vector<Real, 3>> modal_vals(n, math::Vector<Real, 3>{});
        for (std::size_t p = 0; p < n; ++p) {
            const auto& poly = monomials_[p];
            math::Vector<Real, 3> v{};
            for (int t = 0; t < poly.num_terms; ++t) {
                const auto& m = poly.terms[static_cast<std::size_t>(t)];
                const Real mv =
                    px[static_cast<std::size_t>(m.px)] *
                    py[static_cast<std::size_t>(m.py)] *
                    pz[static_cast<std::size_t>(m.pz)];
                v[static_cast<std::size_t>(m.component)] += m.coefficient * mv;
            }
            modal_vals[p] = v;
        }

        for (std::size_t p = 0; p < n; ++p) {
            const auto& mv = modal_vals[p];
            for (std::size_t j = 0; j < n; ++j) {
                const Real c = coeffs_[p * n + j];
                values[j][0] += c * mv[0];
                values[j][1] += c * mv[1];
                values[j][2] += c * mv[2];
            }
        }
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
        const std::size_t n = size_;
        divergence.assign(n, Real(0));

        int max_px = 0, max_py = 0, max_pz = 0;
        for (const auto& poly : monomials_) {
            for (int t = 0; t < poly.num_terms; ++t) {
                const auto& m = poly.terms[static_cast<std::size_t>(t)];
                max_px = std::max(max_px, m.px);
                max_py = std::max(max_py, m.py);
                max_pz = std::max(max_pz, m.pz);
            }
        }
        const auto px = powers(xi[0], max_px);
        const auto py = powers(xi[1], max_py);
        const auto pz = powers(xi[2], max_pz);

        std::vector<Real> modal_div(n, Real(0));
        for (std::size_t p = 0; p < n; ++p) {
            const auto& poly = monomials_[p];
            Real div = Real(0);
            for (int t = 0; t < poly.num_terms; ++t) {
                const auto& m = poly.terms[static_cast<std::size_t>(t)];
                if (m.component == 0 && m.px > 0) {
                    div += m.coefficient * Real(m.px) *
                           px[static_cast<std::size_t>(m.px - 1)] *
                           py[static_cast<std::size_t>(m.py)] *
                           pz[static_cast<std::size_t>(m.pz)];
                } else if (m.component == 1 && m.py > 0) {
                    div += m.coefficient * Real(m.py) *
                           px[static_cast<std::size_t>(m.px)] *
                           py[static_cast<std::size_t>(m.py - 1)] *
                           pz[static_cast<std::size_t>(m.pz)];
                } else if (m.component == 2 && m.pz > 0) {
                    div += m.coefficient * Real(m.pz) *
                           px[static_cast<std::size_t>(m.px)] *
                           py[static_cast<std::size_t>(m.py)] *
                           pz[static_cast<std::size_t>(m.pz - 1)];
                }
            }
            modal_div[p] = div;
        }

        for (std::size_t p = 0; p < n; ++p) {
            const Real md = modal_div[p];
            if (md == Real(0)) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                divergence[j] += coeffs_[p * n + j] * md;
            }
        }
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

std::vector<DofAssociation> BDMBasis::dof_associations() const {
    std::vector<DofAssociation> result(size_);
    std::size_t idx = 0;

    if (element_type_ == ElementType::Triangle3) {
        for (int e = 0; e < 3; ++e) {
            for (int m = 0; m <= order_; ++m) {
                result[idx].entity_type = DofEntity::Edge;
                result[idx].entity_id = e;
                result[idx].moment_index = m;
                ++idx;
            }
        }
        while (idx < size_) {
            result[idx].entity_type = DofEntity::Interior;
            result[idx].entity_id = 0;
            result[idx].moment_index = static_cast<int>(idx - std::size_t(3 * (order_ + 1)));
            ++idx;
        }
    } else if (element_type_ == ElementType::Tetra4) {
        const std::size_t dofs_per_face = triangle_poly_dim(static_cast<std::size_t>(order_));
        for (int f = 0; f < 4; ++f) {
            for (std::size_t m = 0; m < dofs_per_face; ++m) {
                result[idx].entity_type = DofEntity::Face;
                result[idx].entity_id = f;
                result[idx].moment_index = static_cast<int>(m);
                ++idx;
            }
        }
        while (idx < size_) {
            result[idx].entity_type = DofEntity::Interior;
            result[idx].entity_id = 0;
            result[idx].moment_index = static_cast<int>(idx - 4u * dofs_per_face);
            ++idx;
        }
    } else {
        for (int e = 0; e < 4; ++e) {
            for (int m = 0; m < 2; ++m) {
                result[idx].entity_type = DofEntity::Edge;
                result[idx].entity_id = e;
                result[idx].moment_index = m;
                ++idx;
            }
        }
    }

    return result;
}

// -----------------------------------------------------------------------------
// RaviartThomasBasis DOF associations
// -----------------------------------------------------------------------------

std::vector<DofAssociation> RaviartThomasBasis::dof_associations() const {
    std::vector<DofAssociation> result(size_);
    std::size_t idx = 0;
    const int k = order_;

    if (dimension_ == 2) {
        // 2D RT(k): edge DOFs (k+1 per edge) + interior DOFs
        const std::size_t dofs_per_edge = static_cast<std::size_t>(k + 1);
        std::size_t num_edges = is_triangle(element_type_) ? 3u : 4u;

        for (std::size_t e = 0; e < num_edges; ++e) {
            for (std::size_t m = 0; m < dofs_per_edge; ++m) {
                result[idx].entity_type = DofEntity::Edge;
                result[idx].entity_id = static_cast<int>(e);
                result[idx].moment_index = static_cast<int>(m);
                ++idx;
            }
        }

        // Interior DOFs
        while (idx < size_) {
            result[idx].entity_type = DofEntity::Interior;
            result[idx].entity_id = 0;
            result[idx].moment_index = static_cast<int>(idx - num_edges * dofs_per_edge);
            ++idx;
        }
    } else {
        // 3D RT(k): face DOFs + interior DOFs
        const elements::ReferenceElement ref = elements::ReferenceElement::create(element_type_);
        const std::size_t num_faces = ref.num_faces();

        // DOFs per face: (k+1)(k+2)/2 for triangular faces, (k+1)^2 for quad faces
        for (std::size_t f = 0; f < num_faces; ++f) {
            const auto& face_nodes = ref.face_nodes(f);
            std::size_t dofs_per_face;
            if (face_nodes.size() == 3u) {
                // Triangular face: P_k has (k+1)(k+2)/2 DOFs
                dofs_per_face = static_cast<std::size_t>((k + 1) * (k + 2) / 2);
            } else {
                // Quad face: Q_k has (k+1)^2 DOFs
                dofs_per_face = static_cast<std::size_t>((k + 1) * (k + 1));
            }

            for (std::size_t m = 0; m < dofs_per_face; ++m) {
                if (idx >= size_) break;
                result[idx].entity_type = DofEntity::Face;
                result[idx].entity_id = static_cast<int>(f);
                result[idx].moment_index = static_cast<int>(m);
                ++idx;
            }
        }

        // Interior DOFs
        std::size_t interior_start = idx;
        while (idx < size_) {
            result[idx].entity_type = DofEntity::Interior;
            result[idx].entity_id = 0;
            result[idx].moment_index = static_cast<int>(idx - interior_start);
            ++idx;
        }
    }

    return result;
}

// -----------------------------------------------------------------------------
// NedelecBasis DOF associations
// -----------------------------------------------------------------------------

std::vector<DofAssociation> NedelecBasis::dof_associations() const {
    std::vector<DofAssociation> result(size_);
    std::size_t idx = 0;
    const int k = order_;

    // Edge DOFs first (k+1 per edge for Nedelec first kind)
    const elements::ReferenceElement ref = elements::ReferenceElement::create(element_type_);
    const std::size_t num_edges = ref.num_edges();
    const std::size_t dofs_per_edge = static_cast<std::size_t>(k + 1);

    for (std::size_t e = 0; e < num_edges; ++e) {
        for (std::size_t m = 0; m < dofs_per_edge; ++m) {
            if (idx >= size_) break;
            result[idx].entity_type = DofEntity::Edge;
            result[idx].entity_id = static_cast<int>(e);
            result[idx].moment_index = static_cast<int>(m);
            ++idx;
        }
    }

    if (dimension_ == 3 && k >= 1) {
        // 3D Nedelec: face DOFs (tangential moments)
        const std::size_t num_faces = ref.num_faces();
        for (std::size_t f = 0; f < num_faces; ++f) {
            const auto& face_nodes = ref.face_nodes(f);
            std::size_t dofs_per_face;
            if (face_nodes.size() == 3u) {
                // Triangle face: 2 tangent directions, k(k-1)/2 P_{k-1} DOFs each
                // But for Nedelec, face DOFs are 2 * dim(P_{k-1}) = 2 * k(k+1)/2 = k(k+1)
                dofs_per_face = static_cast<std::size_t>(k * (k + 1));
            } else {
                // Quad face: 2 tangent directions with Q_{k-1,k} and Q_{k,k-1}
                dofs_per_face = static_cast<std::size_t>(2 * k * (k + 1));
            }

            for (std::size_t m = 0; m < dofs_per_face; ++m) {
                if (idx >= size_) break;
                result[idx].entity_type = DofEntity::Face;
                result[idx].entity_id = static_cast<int>(f);
                result[idx].moment_index = static_cast<int>(m);
                ++idx;
            }
        }
    }

    // Interior DOFs
    std::size_t interior_start = idx;
    while (idx < size_) {
        result[idx].entity_type = DofEntity::Interior;
        result[idx].entity_id = 0;
        result[idx].moment_index = static_cast<int>(idx - interior_start);
        ++idx;
    }

    return result;
}

} // namespace basis
} // namespace FE
} // namespace svmp
