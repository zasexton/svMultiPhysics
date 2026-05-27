/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "VectorBasisDirectSeeds.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#ifdef FE_CHECK_ARG
#undef FE_CHECK_ARG
#endif
#define FE_CHECK_ARG(condition, message) BASIS_CHECK_CONSTRUCTION((condition), (message))

namespace svmp {
namespace FE {
namespace basis {

using Vec3 = math::Vector<Real, 3>;

namespace {

class Poly3 {
public:
    using Exponents = std::array<int, 3>;
    using TermMap = std::map<Exponents, Real>;

    Poly3() = default;

    Poly3(Real constant) {
        if (constant != Real(0)) {
            terms_[{0, 0, 0}] = constant;
        }
    }

    static Poly3 variable(int axis) {
        Poly3 poly;
        Exponents exponents{0, 0, 0};
        exponents[static_cast<std::size_t>(axis)] = 1;
        poly.terms_[exponents] = Real(1);
        return poly;
    }

    const TermMap& terms() const noexcept { return terms_; }

    bool is_constant() const noexcept {
        return terms_.empty() ||
               (terms_.size() == 1u && terms_.begin()->first == Exponents{0, 0, 0});
    }

    Real constant_value() const noexcept {
        if (terms_.empty()) {
            return Real(0);
        }
        return terms_.begin()->second;
    }

    void add_term(const Exponents& exponents, Real coeff) {
        if (coeff == Real(0)) {
            return;
        }
        auto& slot = terms_[exponents];
        slot += coeff;
        if (slot == Real(0)) {
            terms_.erase(exponents);
        }
    }

private:
    TermMap terms_;
};

inline Poly3 operator+(Poly3 a, const Poly3& b) {
    for (const auto& [exponents, coeff] : b.terms()) {
        a.add_term(exponents, coeff);
    }
    return a;
}

inline Poly3 operator-(Poly3 a, const Poly3& b) {
    for (const auto& [exponents, coeff] : b.terms()) {
        a.add_term(exponents, -coeff);
    }
    return a;
}

inline Poly3 operator-(const Poly3& a) {
    Poly3 result;
    for (const auto& [exponents, coeff] : a.terms()) {
        result.add_term(exponents, -coeff);
    }
    return result;
}

inline Poly3 operator*(const Poly3& a, const Poly3& b) {
    Poly3 result;
    for (const auto& [a_exp, a_coeff] : a.terms()) {
        for (const auto& [b_exp, b_coeff] : b.terms()) {
            Poly3::Exponents exponents{
                a_exp[0] + b_exp[0],
                a_exp[1] + b_exp[1],
                a_exp[2] + b_exp[2]};
            result.add_term(exponents, a_coeff * b_coeff);
        }
    }
    return result;
}

inline Poly3 operator/(const Poly3& a, const Poly3& b) {
    BASIS_CHECK_EVAL(b.is_constant() && b.constant_value() != Real(0),
                     "VectorBasisDirectSeeds: polynomial division requires a nonzero constant denominator");
    const Real inv = Real(1) / b.constant_value();
    Poly3 result;
    for (const auto& [exponents, coeff] : a.terms()) {
        result.add_term(exponents, coeff * inv);
    }
    return result;
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

struct DirectSeedTerm {
    std::uint16_t dof{0u};
    std::uint8_t component{0u};
    std::array<std::uint8_t, 3> exponents{0u, 0u, 0u};
    Real coefficient{0};
};

struct DirectSeedJacobianTerm {
    std::uint16_t dof{0u};
    std::uint8_t component{0u};
    std::uint8_t derivative_axis{0u};
    std::array<std::uint8_t, 3> exponents{0u, 0u, 0u};
    Real coefficient{0};
};

struct DirectSeedTable {
    std::size_t size{0u};
    std::vector<DirectSeedTerm> value_terms;
    std::vector<DirectSeedJacobianTerm> jacobian_terms;
    std::vector<DirectSeedTerm> divergence_terms;
    std::vector<DirectSeedTerm> curl_terms;
    std::array<std::uint8_t, 3> max_exponents{0u, 0u, 0u};
};

constexpr std::size_t kDirectSeedMaxStoredExponent = 16u;

using DirectSeedPowers =
    std::array<std::array<Real, kDirectSeedMaxStoredExponent + 1u>, 3>;

void record_max_exponents(DirectSeedTable& table,
                          const std::array<std::uint8_t, 3>& exponents) {
    for (std::size_t axis = 0; axis < 3u; ++axis) {
        table.max_exponents[axis] = std::max(table.max_exponents[axis], exponents[axis]);
    }
}

DirectSeedPowers make_direct_seed_powers(const DirectSeedTable& table,
                                         const Vec3& xi) {
    DirectSeedPowers powers{};
    for (std::size_t axis = 0; axis < 3u; ++axis) {
        powers[axis][0] = Real(1);
        for (std::uint8_t exponent = 1u; exponent <= table.max_exponents[axis]; ++exponent) {
            powers[axis][exponent] = powers[axis][exponent - 1u] * xi[axis];
        }
    }
    return powers;
}

Real evaluate_monomial(const DirectSeedTerm& term,
                       const DirectSeedPowers& powers) {
    return term.coefficient *
           powers[0][term.exponents[0]] *
           powers[1][term.exponents[1]] *
           powers[2][term.exponents[2]];
}

Real evaluate_monomial(const DirectSeedJacobianTerm& term,
                       const DirectSeedPowers& powers) {
    return term.coefficient *
           powers[0][term.exponents[0]] *
           powers[1][term.exponents[1]] *
           powers[2][term.exponents[2]];
}

bool make_derivative_term(const DirectSeedTerm& term,
                          std::size_t axis,
                          DirectSeedTerm& derivative) {
    const std::uint8_t exponent = term.exponents[axis];
    if (exponent == 0u) {
        return false;
    }
    derivative = term;
    --derivative.exponents[axis];
    derivative.coefficient *= static_cast<Real>(exponent);
    return true;
}

template <typename EvalFn>
DirectSeedTable build_direct_seed_table(EvalFn&& eval) {
    std::vector<DirectVec3<Poly3>> polynomial_values;
    eval(make_direct_vec3(Poly3::variable(0), Poly3::variable(1), Poly3::variable(2)),
         polynomial_values);

    DirectSeedTable table;
    table.size = polynomial_values.size();
    for (std::size_t dof = 0; dof < polynomial_values.size(); ++dof) {
        BASIS_CHECK_EVAL(dof <= static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()),
                         "VectorBasisDirectSeeds: direct seed table exceeds compact DOF index range");
        for (std::size_t component = 0; component < 3u; ++component) {
            for (const auto& [exponents, coeff] : polynomial_values[dof][component].terms()) {
                DirectSeedTerm term{};
                term.dof = static_cast<std::uint16_t>(dof);
                term.component = static_cast<std::uint8_t>(component);
                for (std::size_t axis = 0; axis < 3u; ++axis) {
                    BASIS_CHECK_EVAL(exponents[axis] >= 0 && exponents[axis] <= 255,
                                     "VectorBasisDirectSeeds: monomial exponent exceeds compact table range");
                    term.exponents[axis] = static_cast<std::uint8_t>(exponents[axis]);
                    BASIS_CHECK_EVAL(term.exponents[axis] <= kDirectSeedMaxStoredExponent,
                                     "VectorBasisDirectSeeds: monomial exponent exceeds runtime power table range");
                }
                term.coefficient = coeff;
                table.value_terms.push_back(term);
                record_max_exponents(table, term.exponents);

                for (std::size_t axis = 0; axis < 3u; ++axis) {
                    DirectSeedTerm derivative{};
                    if (!make_derivative_term(term, axis, derivative)) {
                        continue;
                    }

                    DirectSeedJacobianTerm jacobian_term{};
                    jacobian_term.dof = derivative.dof;
                    jacobian_term.component = derivative.component;
                    jacobian_term.derivative_axis = static_cast<std::uint8_t>(axis);
                    jacobian_term.exponents = derivative.exponents;
                    jacobian_term.coefficient = derivative.coefficient;
                    table.jacobian_terms.push_back(jacobian_term);
                    record_max_exponents(table, derivative.exponents);

                    if (axis == component) {
                        table.divergence_terms.push_back(derivative);
                    }

                    if (component == 0u && axis == 2u) {
                        derivative.component = 1u;
                        table.curl_terms.push_back(derivative);
                    } else if (component == 0u && axis == 1u) {
                        derivative.component = 2u;
                        derivative.coefficient = -derivative.coefficient;
                        table.curl_terms.push_back(derivative);
                    } else if (component == 1u && axis == 2u) {
                        derivative.component = 0u;
                        derivative.coefficient = -derivative.coefficient;
                        table.curl_terms.push_back(derivative);
                    } else if (component == 1u && axis == 0u) {
                        derivative.component = 2u;
                        table.curl_terms.push_back(derivative);
                    } else if (component == 2u && axis == 1u) {
                        derivative.component = 0u;
                        table.curl_terms.push_back(derivative);
                    } else if (component == 2u && axis == 0u) {
                        derivative.component = 1u;
                        derivative.coefficient = -derivative.coefficient;
                        table.curl_terms.push_back(derivative);
                    }
                    record_max_exponents(table, derivative.exponents);
                }
            }
        }
    }
    return table;
}

void eval_direct_table_values(const DirectSeedTable& table,
                              const Vec3& xi,
                              std::vector<Vec3>& values) {
    values.assign(table.size, Vec3{});
    const DirectSeedPowers powers = make_direct_seed_powers(table, xi);
    for (const DirectSeedTerm& term : table.value_terms) {
        values[term.dof][term.component] += evaluate_monomial(term, powers);
    }
}

void eval_direct_table_jacobians(const DirectSeedTable& table,
                                 const Vec3& xi,
                                 std::vector<VectorJacobian>& jacobians) {
    jacobians.assign(table.size, VectorJacobian{});
    const DirectSeedPowers powers = make_direct_seed_powers(table, xi);
    for (const DirectSeedJacobianTerm& term : table.jacobian_terms) {
        jacobians[term.dof](term.component, term.derivative_axis) +=
            evaluate_monomial(term, powers);
    }
}

void eval_direct_table_divergence(const DirectSeedTable& table,
                                  const Vec3& xi,
                                  std::vector<Real>& divergence) {
    divergence.assign(table.size, Real(0));
    const DirectSeedPowers powers = make_direct_seed_powers(table, xi);
    for (const DirectSeedTerm& term : table.divergence_terms) {
        divergence[term.dof] += evaluate_monomial(term, powers);
    }
}

void eval_direct_table_curl(const DirectSeedTable& table,
                            const Vec3& xi,
                            std::vector<Vec3>& curl) {
    curl.assign(table.size, Vec3{});
    const DirectSeedPowers powers = make_direct_seed_powers(table, xi);
    for (const DirectSeedTerm& term : table.curl_terms) {
        curl[term.dof][term.component] += evaluate_monomial(term, powers);
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

template <typename Scalar>
inline void eval_wedge_rt1_direct_impl(const DirectVec3<Scalar>& xi,
                                       std::vector<DirectVec3<Scalar>>& values) {
    using ValueVec = DirectVec3<Scalar>;

    const auto x = xi[0];
    const auto y = xi[1];
    const auto z = xi[2];

    // Barycentric coordinates for the triangular cross-section
    const auto L0 = Real(1) - x - y;  // opposite to edge from v1 to v2
    const auto L1 = x;
    const auto L2 = y;

    // Linear z-selectors
    const auto zb = (Real(1) - z) * Real(0.5);  // 1 at z=-1, 0 at z=+1
    const auto zt = (Real(1) + z) * Real(0.5);  // 0 at z=-1, 1 at z=+1

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
    values[idx++] = ValueVec{Real(0), Real(0), zb * (Real(-2))};
    // Face DOF 1: test = x (linear in x)
    values[idx++] = ValueVec{Real(0), Real(0), zb * (Real(-6) * x + Real(2))};
    // Face DOF 2: test = y (linear in y)
    values[idx++] = ValueVec{Real(0), Real(0), zb * (Real(-6) * y + Real(2))};

    // ==========================================================================
    // Top triangular face (z=+1): 3 DOFs - P_1 normal flux moments
    // Normal n = (0, 0, +1), so v.n = v_z
    // ==========================================================================
    // Face DOF 3: test = 1
    values[idx++] = ValueVec{Real(0), Real(0), zt * Real(2)};
    // Face DOF 4: test = x
    values[idx++] = ValueVec{Real(0), Real(0), zt * (Real(6) * x - Real(2))};
    // Face DOF 5: test = y
    values[idx++] = ValueVec{Real(0), Real(0), zt * (Real(6) * y - Real(2))};

    // ==========================================================================
    // Quad face 2 (y=0): 4 DOFs - Q_1 normal flux moments
    // Normal n = (0, -1, 0), so v.n = -v_y
    // Face parameterized by (x, z) in [0,1] x [-1,1]
    // Q_1 test functions: {1, x, z, x*z} (mapped to face coords)
    // ==========================================================================
    // Use y-localization: (1-y) gives 1 on face, 0 on opposite edge
    values[idx++] = ValueVec{Real(0), (y - Real(1)), Real(0)}; // DOF 6: test = 1
    values[idx++] = ValueVec{Real(0), (y - Real(1)) * (Real(3) * x - Real(1)), Real(0)}; // DOF 7: test = x
    values[idx++] = ValueVec{Real(0), (y - Real(1)) * z, Real(0)}; // DOF 8: test = z
    values[idx++] = ValueVec{Real(0), (y - Real(1)) * (Real(3) * x - Real(1)) * z, Real(0)}; // DOF 9: test = x*z

    // ==========================================================================
    // Quad face 3 (x=0): 4 DOFs - Q_1 normal flux moments
    // Normal n = (-1, 0, 0), so v.n = -v_x
    // Face parameterized by (y, z) in [0,1] x [-1,1]
    // ==========================================================================
    values[idx++] = ValueVec{(x - Real(1)), Real(0), Real(0)}; // DOF 10: test = 1
    values[idx++] = ValueVec{(x - Real(1)) * (Real(3) * y - Real(1)), Real(0), Real(0)}; // DOF 11: test = y
    values[idx++] = ValueVec{(x - Real(1)) * z, Real(0), Real(0)}; // DOF 12: test = z
    values[idx++] = ValueVec{(x - Real(1)) * (Real(3) * y - Real(1)) * z, Real(0), Real(0)}; // DOF 13: test = y*z

    // ==========================================================================
    // Quad face 4 (x+y=1): 4 DOFs - Q_1 normal flux moments
    // Normal n = (1/sqrt(2), 1/sqrt(2), 0), so v.n = (v_x + v_y)/sqrt(2)
    // Face parameterized by t = x (so y = 1-t), z in [-1,1]
    // Use localization: (x+y) - (1-x-y) = 2(x+y) - 1
    // On face x+y=1: localization = 1
    // On opposite edge (x=0, y=0): localization = -1
    // ==========================================================================
    const auto loc4 = x + y;  // 1 on face, 0 on opposite vertex
    // v.n needs to integrate correctly against Q_1 tests
    // Use v = (loc4 * f, loc4 * f, 0) so v.n = sqrt(2) * loc4 * f
    values[idx++] = ValueVec{loc4, loc4, Real(0)}; // DOF 14: test = 1
    values[idx++] = ValueVec{loc4 * (Real(3) * x - Real(1)), loc4 * (Real(3) * x - Real(1)), Real(0)}; // DOF 15: test = s (face param)
    values[idx++] = ValueVec{loc4 * z, loc4 * z, Real(0)}; // DOF 16: test = z
    values[idx++] = ValueVec{loc4 * (Real(3) * x - Real(1)) * z, loc4 * (Real(3) * x - Real(1)) * z, Real(0)}; // DOF 17: test = s*z

    // ==========================================================================
    // Interior DOFs: 6 DOFs
    // These are bubble functions with zero normal flux on all faces
    // Interior test space: P_0(x,y) x P_1(z) for each of 3 components
    // But we need divergence-compatible functions
    // Use: v = bubble(x,y,z) * constant_vector
    // Bubble = L0*L1*L2 * (1-z^2) = x*(1-x-y)*y*(1-z^2)
    // ==========================================================================
    const auto bubble_xy = L0 * L1 * L2;  // = x*(1-x-y)*y, zero on all edges of triangle
    const auto bubble_z = (Real(1) - z * z);  // zero at z = +/- 1
    const auto bubble = bubble_xy * bubble_z;

    // 6 interior DOFs: test against {e_x, e_y, e_z, z*e_x, z*e_y, z*e_z}
    values[idx++] = ValueVec{bubble * Real(60), Real(0), Real(0)}; // DOF 18
    values[idx++] = ValueVec{Real(0), bubble * Real(60), Real(0)}; // DOF 19
    values[idx++] = ValueVec{Real(0), Real(0), bubble * Real(60)}; // DOF 20
    values[idx++] = ValueVec{bubble * z * Real(180), Real(0), Real(0)}; // DOF 21
    values[idx++] = ValueVec{Real(0), bubble * z * Real(180), Real(0)}; // DOF 22
    values[idx++] = ValueVec{Real(0), Real(0), bubble * z * Real(180)}; // DOF 23
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

template <typename Scalar>
inline void eval_pyramid_rt1_direct_impl(const DirectVec3<Scalar>& xi,
                                         std::vector<DirectVec3<Scalar>>& values) {
    using ValueVec = DirectVec3<Scalar>;

    const auto x = xi[0];
    const auto y = xi[1];
    const auto z = xi[2];

    // For pyramid, z in [0,1], base at z=0, apex at z=1
    // Base spans [-1,1] x [-1,1] at z=0

    // Vertex-based coordinates for the base quad
    const auto zc = Real(1) - z;  // complement of z, 1 at base, 0 at apex

    values.resize(19);
    std::size_t idx = 0;

    // ==========================================================================
    // Quad base face (z=0): 4 DOFs - Q_1 normal flux moments
    // Normal n = (0,0,-1), v.n = -v_z
    // Q_1 test functions on base: {1, x, y, xy}
    // ==========================================================================
    values[idx++] = ValueVec{Real(0), Real(0), -zc}; // DOF 0: test = 1
    values[idx++] = ValueVec{Real(0), Real(0), -zc * x * Real(3)}; // DOF 1: test = x
    values[idx++] = ValueVec{Real(0), Real(0), -zc * y * Real(3)}; // DOF 2: test = y
    values[idx++] = ValueVec{Real(0), Real(0), -zc * x * y * Real(9)}; // DOF 3: test = xy

    // ==========================================================================
    // Triangular face 1 (front, y = -1+z at apex): 3 DOFs
    // Vertices: v0=(-1,-1,0), v1=(1,-1,0), v4=(0,0,1)
    // Outward normal points in -y direction
    // ==========================================================================
    const auto loc_f1 = Real(1) + y - z;  // 1 on face 1, decreases away from it
    values[idx++] = ValueVec{Real(0), -loc_f1, Real(0)}; // DOF 4: test = 1
    values[idx++] = ValueVec{Real(0), -loc_f1 * x * Real(3), Real(0)}; // DOF 5: test = x (face param)
    values[idx++] = ValueVec{Real(0), -loc_f1 * z * Real(3), Real(0)}; // DOF 6: test = z (face param)

    // ==========================================================================
    // Triangular face 2 (right, x = 1-z at apex): 3 DOFs
    // Vertices: v1=(1,-1,0), v2=(1,1,0), v4=(0,0,1)
    // Outward normal points in +x direction
    // ==========================================================================
    const auto loc_f2 = Real(1) - x - z;  // 1 on face 2
    values[idx++] = ValueVec{-loc_f2, Real(0), Real(0)}; // DOF 7: test = 1
    values[idx++] = ValueVec{-loc_f2 * y * Real(3), Real(0), Real(0)}; // DOF 8: test = y
    values[idx++] = ValueVec{-loc_f2 * z * Real(3), Real(0), Real(0)}; // DOF 9: test = z

    // ==========================================================================
    // Triangular face 3 (back, y = 1-z at apex): 3 DOFs
    // Vertices: v2=(1,1,0), v3=(-1,1,0), v4=(0,0,1)
    // Outward normal points in +y direction
    // ==========================================================================
    const auto loc_f3 = Real(1) - y - z;  // 1 on face 3
    values[idx++] = ValueVec{Real(0), -loc_f3, Real(0)}; // DOF 10: test = 1
    values[idx++] = ValueVec{Real(0), -loc_f3 * x * Real(3), Real(0)}; // DOF 11: test = x
    values[idx++] = ValueVec{Real(0), -loc_f3 * z * Real(3), Real(0)}; // DOF 12: test = z

    // ==========================================================================
    // Triangular face 4 (left, x = -1+z at apex): 3 DOFs
    // Vertices: v3=(-1,1,0), v0=(-1,-1,0), v4=(0,0,1)
    // Outward normal points in -x direction
    // ==========================================================================
    const auto loc_f4 = Real(1) + x - z;  // 1 on face 4
    values[idx++] = ValueVec{loc_f4, Real(0), Real(0)}; // DOF 13: test = 1
    values[idx++] = ValueVec{loc_f4 * y * Real(3), Real(0), Real(0)}; // DOF 14: test = y
    values[idx++] = ValueVec{loc_f4 * z * Real(3), Real(0), Real(0)}; // DOF 15: test = z

    // ==========================================================================
    // Interior DOFs: 3 DOFs
    // Bubble function that vanishes on all faces
    // ==========================================================================
    // Interior bubble: product of all face localizations
    const auto bubble = zc * loc_f1 * loc_f2 * loc_f3 * loc_f4;
    values[idx++] = ValueVec{bubble * Real(120), Real(0), Real(0)}; // DOF 16
    values[idx++] = ValueVec{Real(0), bubble * Real(120), Real(0)}; // DOF 17
    values[idx++] = ValueVec{Real(0), Real(0), bubble * Real(120)}; // DOF 18
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
        values[idx++] = Vec3{zb * (Real(1) - y) * leg, Real(0), Real(0)};
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

const DirectSeedTable& wedge_rt1_table() {
    static const DirectSeedTable table =
        build_direct_seed_table(eval_wedge_rt1_direct_impl<Poly3>);
    return table;
}

const DirectSeedTable& wedge_rt2_table() {
    static const DirectSeedTable table =
        build_direct_seed_table(eval_wedge_rt2_direct_impl<Poly3>);
    return table;
}

const DirectSeedTable& pyramid_rt1_table() {
    static const DirectSeedTable table =
        build_direct_seed_table(eval_pyramid_rt1_direct_impl<Poly3>);
    return table;
}

const DirectSeedTable& pyramid_rt2_table() {
    static const DirectSeedTable table =
        build_direct_seed_table(eval_pyramid_rt2_direct_impl<Poly3>);
    return table;
}

const DirectSeedTable& wedge_nd1_table() {
    static const DirectSeedTable table =
        build_direct_seed_table(eval_wedge_nd1_direct_impl<Poly3>);
    return table;
}

const DirectSeedTable& wedge_nd2_table() {
    static const DirectSeedTable table =
        build_direct_seed_table(eval_wedge_nd2_direct_impl<Poly3>);
    return table;
}

const DirectSeedTable& pyramid_nd1_table() {
    static const DirectSeedTable table =
        build_direct_seed_table(eval_pyramid_nd1_direct_impl<Poly3>);
    return table;
}

const DirectSeedTable& pyramid_nd2_table() {
    static const DirectSeedTable table =
        build_direct_seed_table(eval_pyramid_nd2_direct_impl<Poly3>);
    return table;
}

} // namespace

namespace detail {
namespace vector_direct {

void eval_wedge_rt1_values(const Vec3& xi, std::vector<Vec3>& values) {
    eval_direct_table_values(wedge_rt1_table(), xi, values);
}

void eval_wedge_rt1_divergence(const Vec3& xi, std::vector<Real>& divergence) {
    eval_direct_table_divergence(wedge_rt1_table(), xi, divergence);
}

void eval_wedge_rt1_jacobians(const Vec3& xi, std::vector<VectorJacobian>& jacobians) {
    eval_direct_table_jacobians(wedge_rt1_table(), xi, jacobians);
}

void eval_wedge_rt2_values(const Vec3& xi, std::vector<Vec3>& values) {
    eval_direct_table_values(wedge_rt2_table(), xi, values);
}

void eval_wedge_rt2_divergence(const Vec3& xi, std::vector<Real>& divergence) {
    eval_direct_table_divergence(wedge_rt2_table(), xi, divergence);
}

void eval_wedge_rt2_jacobians(const Vec3& xi, std::vector<VectorJacobian>& jacobians) {
    eval_direct_table_jacobians(wedge_rt2_table(), xi, jacobians);
}

void eval_pyramid_rt1_values(const Vec3& xi, std::vector<Vec3>& values) {
    eval_direct_table_values(pyramid_rt1_table(), xi, values);
}

void eval_pyramid_rt1_divergence(const Vec3& xi, std::vector<Real>& divergence) {
    eval_direct_table_divergence(pyramid_rt1_table(), xi, divergence);
}

void eval_pyramid_rt1_jacobians(const Vec3& xi, std::vector<VectorJacobian>& jacobians) {
    eval_direct_table_jacobians(pyramid_rt1_table(), xi, jacobians);
}

void eval_pyramid_rt2_values(const Vec3& xi, std::vector<Vec3>& values) {
    eval_direct_table_values(pyramid_rt2_table(), xi, values);
}

void eval_pyramid_rt2_divergence(const Vec3& xi, std::vector<Real>& divergence) {
    eval_direct_table_divergence(pyramid_rt2_table(), xi, divergence);
}

void eval_pyramid_rt2_jacobians(const Vec3& xi, std::vector<VectorJacobian>& jacobians) {
    eval_direct_table_jacobians(pyramid_rt2_table(), xi, jacobians);
}

void eval_wedge_nd1_values(const Vec3& xi, std::vector<Vec3>& values) {
    eval_direct_table_values(wedge_nd1_table(), xi, values);
}

void eval_wedge_nd1_jacobians(const Vec3& xi, std::vector<VectorJacobian>& jacobians) {
    eval_direct_table_jacobians(wedge_nd1_table(), xi, jacobians);
}

void eval_wedge_nd1_curl(const Vec3& xi, std::vector<Vec3>& curl) {
    eval_direct_table_curl(wedge_nd1_table(), xi, curl);
}

void eval_wedge_nd2_values(const Vec3& xi, std::vector<Vec3>& values) {
    eval_direct_table_values(wedge_nd2_table(), xi, values);
}

void eval_wedge_nd2_jacobians(const Vec3& xi, std::vector<VectorJacobian>& jacobians) {
    eval_direct_table_jacobians(wedge_nd2_table(), xi, jacobians);
}

void eval_wedge_nd2_curl(const Vec3& xi, std::vector<Vec3>& curl) {
    eval_direct_table_curl(wedge_nd2_table(), xi, curl);
}

void eval_pyramid_nd1_values(const Vec3& xi, std::vector<Vec3>& values) {
    eval_direct_table_values(pyramid_nd1_table(), xi, values);
}

void eval_pyramid_nd1_jacobians(const Vec3& xi, std::vector<VectorJacobian>& jacobians) {
    eval_direct_table_jacobians(pyramid_nd1_table(), xi, jacobians);
}

void eval_pyramid_nd1_curl(const Vec3& xi, std::vector<Vec3>& curl) {
    eval_direct_table_curl(pyramid_nd1_table(), xi, curl);
}

void eval_pyramid_nd2_values(const Vec3& xi, std::vector<Vec3>& values) {
    eval_direct_table_values(pyramid_nd2_table(), xi, values);
}

void eval_pyramid_nd2_jacobians(const Vec3& xi, std::vector<VectorJacobian>& jacobians) {
    eval_direct_table_jacobians(pyramid_nd2_table(), xi, jacobians);
}

void eval_pyramid_nd2_curl(const Vec3& xi, std::vector<Vec3>& curl) {
    eval_direct_table_curl(pyramid_nd2_table(), xi, curl);
}

} // namespace vector_direct
} // namespace detail


} // namespace basis
} // namespace FE
} // namespace svmp
