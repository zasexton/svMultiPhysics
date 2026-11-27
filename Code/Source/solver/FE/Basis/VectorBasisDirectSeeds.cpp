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

#define SVMP_DEFINE_DIRECT_SEED_TABLE(name, impl) \
const DirectSeedTable& name##_table() { \
    static const DirectSeedTable table = build_direct_seed_table(impl); \
    return table; \
}

SVMP_DEFINE_DIRECT_SEED_TABLE(pyramid_rt1, eval_pyramid_rt1_direct_impl<Poly3>)
SVMP_DEFINE_DIRECT_SEED_TABLE(pyramid_rt2, eval_pyramid_rt2_direct_impl<Poly3>)

#undef SVMP_DEFINE_DIRECT_SEED_TABLE

} // namespace

namespace detail {
namespace vector_direct {

#define SVMP_DEFINE_RT_DIRECT_WRAPPERS(name) \
void eval_##name##_values(const Vec3& xi, std::vector<Vec3>& values) { \
    eval_direct_table_values(name##_table(), xi, values); \
} \
void eval_##name##_divergence(const Vec3& xi, std::vector<Real>& divergence) { \
    eval_direct_table_divergence(name##_table(), xi, divergence); \
} \
void eval_##name##_jacobians(const Vec3& xi, std::vector<VectorJacobian>& jacobians) { \
    eval_direct_table_jacobians(name##_table(), xi, jacobians); \
}

SVMP_DEFINE_RT_DIRECT_WRAPPERS(pyramid_rt1)
SVMP_DEFINE_RT_DIRECT_WRAPPERS(pyramid_rt2)

#undef SVMP_DEFINE_RT_DIRECT_WRAPPERS

} // namespace vector_direct
} // namespace detail


} // namespace basis
} // namespace FE
} // namespace svmp
