/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "LagrangeBasis.h"
#include "BasisTraits.h"
#include "BasisTolerance.h"
#include "LagrangeBasisFast.h"
#include "NodeOrderingConventions.h"
#include "LagrangeBasisPyramid.h"
#include "LagrangeBasisSimplex.h"
#include "LagrangeBasisUtility.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace basis {

namespace {

using LagrangeTopology = BasisTopology;

struct LagrangeTopologyTraits {
    LagrangeTopology topology;
    int dimension;
};

struct SimplexExponentHash {
    std::size_t operator()(const std::array<int, 4>& exponents) const noexcept {
        std::size_t seed = 0x9e3779b97f4a7c15ull;
        for (const int exponent : exponents) {
            const auto value = static_cast<std::size_t>(exponent);
            seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
        }
        return seed;
    }
};

template<typename T, std::size_t N>
void assign_array(std::vector<T>& out, const std::array<T, N>& values) {
    out.assign(values.begin(), values.end());
}

bool coordinate_matches_expected(Real coord, Real expected) noexcept {
    return detail::basis_nearly_equal(coord, expected);
}

template<typename FastBasis>
void evaluate_fast_outputs(const math::Vector<Real, 3>& xi,
                           std::vector<Real>* values,
                           std::vector<Gradient>* gradients,
                           std::vector<Hessian>* hessians) {
    if (values != nullptr) {
        std::array<Real, FastBasis::n_dofs> fast_values{};
        FastBasis::evaluate(xi, fast_values);
        assign_array(*values, fast_values);
    }
    if (gradients != nullptr) {
        std::array<Gradient, FastBasis::n_dofs> fast_gradients{};
        FastBasis::evaluate_gradients(xi, fast_gradients);
        assign_array(*gradients, fast_gradients);
    }
    if (hessians != nullptr) {
        std::array<Hessian, FastBasis::n_dofs> fast_hessians{};
        FastBasis::evaluate_hessians(xi, fast_hessians);
        assign_array(*hessians, fast_hessians);
    }
}

template<typename FastBasis>
void evaluate_fast_outputs_to(const math::Vector<Real, 3>& xi,
                              Real* SVMP_RESTRICT values_out,
                              Real* SVMP_RESTRICT gradients_out,
                              Real* SVMP_RESTRICT hessians_out) {
    if (values_out != nullptr) {
        std::array<Real, FastBasis::n_dofs> fast_values{};
        FastBasis::evaluate(xi, fast_values);
        for (std::size_t i = 0; i < fast_values.size(); ++i) {
            values_out[i] = fast_values[i];
        }
    }
    if (gradients_out != nullptr) {
        std::array<Gradient, FastBasis::n_dofs> fast_gradients{};
        FastBasis::evaluate_gradients(xi, fast_gradients);
        for (std::size_t i = 0; i < fast_gradients.size(); ++i) {
            gradients_out[i * 3u + 0u] = fast_gradients[i][0];
            gradients_out[i * 3u + 1u] = fast_gradients[i][1];
            gradients_out[i * 3u + 2u] = fast_gradients[i][2];
        }
    }
    if (hessians_out != nullptr) {
        std::array<Hessian, FastBasis::n_dofs> fast_hessians{};
        FastBasis::evaluate_hessians(xi, fast_hessians);
        for (std::size_t i = 0; i < fast_hessians.size(); ++i) {
            store_hessian(fast_hessians[i], hessians_out + i * 9u);
        }
    }
}

bool evaluate_fixed_lagrange_fast(LagrangeTopology topology,
                                  int order,
                                  const math::Vector<Real, 3>& xi,
                                  std::vector<Real>* values,
                                  std::vector<Gradient>* gradients,
                                  std::vector<Hessian>* hessians) {
    if (order == 1) {
        switch (topology) {
            case LagrangeTopology::Line:
                evaluate_fast_outputs<LagrangeLineFast<1>>(xi, values, gradients, hessians);
                return true;
            case LagrangeTopology::Quadrilateral:
                evaluate_fast_outputs<LagrangeQuadFast<1>>(xi, values, gradients, hessians);
                return true;
            case LagrangeTopology::Hexahedron:
                evaluate_fast_outputs<LagrangeHexFast<1>>(xi, values, gradients, hessians);
                return true;
            case LagrangeTopology::Triangle:
                evaluate_fast_outputs<LagrangeTriFast<1>>(xi, values, gradients, hessians);
                return true;
            case LagrangeTopology::Tetrahedron:
                evaluate_fast_outputs<LagrangeTetFast<1>>(xi, values, gradients, hessians);
                return true;
            default:
                return false;
        }
    }

    if (order == 2) {
        switch (topology) {
            case LagrangeTopology::Line:
                evaluate_fast_outputs<LagrangeLineFast<2>>(xi, values, gradients, hessians);
                return true;
            case LagrangeTopology::Quadrilateral:
                evaluate_fast_outputs<LagrangeQuadFast<2>>(xi, values, gradients, hessians);
                return true;
            case LagrangeTopology::Hexahedron:
                evaluate_fast_outputs<LagrangeHexFast<2>>(xi, values, gradients, hessians);
                return true;
            case LagrangeTopology::Triangle:
                evaluate_fast_outputs<LagrangeTriFast<2>>(xi, values, gradients, hessians);
                return true;
            case LagrangeTopology::Tetrahedron:
                evaluate_fast_outputs<LagrangeTetFast<2>>(xi, values, gradients, hessians);
                return true;
            default:
                return false;
        }
    }

    return false;
}

bool evaluate_fixed_lagrange_fast_to(LagrangeTopology topology,
                                     int order,
                                     const math::Vector<Real, 3>& xi,
                                     Real* SVMP_RESTRICT values_out,
                                     Real* SVMP_RESTRICT gradients_out,
                                     Real* SVMP_RESTRICT hessians_out) {
    if (order == 1) {
        switch (topology) {
            case LagrangeTopology::Line:
                evaluate_fast_outputs_to<LagrangeLineFast<1>>(xi, values_out, gradients_out, hessians_out);
                return true;
            case LagrangeTopology::Quadrilateral:
                evaluate_fast_outputs_to<LagrangeQuadFast<1>>(xi, values_out, gradients_out, hessians_out);
                return true;
            case LagrangeTopology::Hexahedron:
                evaluate_fast_outputs_to<LagrangeHexFast<1>>(xi, values_out, gradients_out, hessians_out);
                return true;
            case LagrangeTopology::Triangle:
                evaluate_fast_outputs_to<LagrangeTriFast<1>>(xi, values_out, gradients_out, hessians_out);
                return true;
            case LagrangeTopology::Tetrahedron:
                evaluate_fast_outputs_to<LagrangeTetFast<1>>(xi, values_out, gradients_out, hessians_out);
                return true;
            default:
                return false;
        }
    }

    if (order == 2) {
        switch (topology) {
            case LagrangeTopology::Line:
                evaluate_fast_outputs_to<LagrangeLineFast<2>>(xi, values_out, gradients_out, hessians_out);
                return true;
            case LagrangeTopology::Quadrilateral:
                evaluate_fast_outputs_to<LagrangeQuadFast<2>>(xi, values_out, gradients_out, hessians_out);
                return true;
            case LagrangeTopology::Hexahedron:
                evaluate_fast_outputs_to<LagrangeHexFast<2>>(xi, values_out, gradients_out, hessians_out);
                return true;
            case LagrangeTopology::Triangle:
                evaluate_fast_outputs_to<LagrangeTriFast<2>>(xi, values_out, gradients_out, hessians_out);
                return true;
            case LagrangeTopology::Tetrahedron:
                evaluate_fast_outputs_to<LagrangeTetFast<2>>(xi, values_out, gradients_out, hessians_out);
                return true;
            default:
                return false;
        }
    }

    return false;
}

template<std::size_t N>
struct AxisMonomialCoefficientTable {
    std::array<Real, N * N> values{};
    std::array<Real, N * (N > 1 ? N - 1 : 0)> first{};
    std::array<Real, N * (N > 2 ? N - 2 : 0)> second{};
};

template<std::size_t N>
constexpr AxisMonomialCoefficientTable<N> make_axis_monomial_coefficient_table() {
    AxisMonomialCoefficientTable<N> table{};
    std::array<Real, N> nodes{};
    constexpr int order = static_cast<int>(N) - 1;
    for (std::size_t i = 0; i < N; ++i) {
        nodes[i] = detail::equispaced_pm_one_coord(static_cast<int>(i), order);
    }

    for (std::size_t i = 0; i < N; ++i) {
        std::array<Real, N> coeffs{};
        std::array<Real, N> next{};
        coeffs[0] = Real(1);
        std::size_t degree = 0;
        for (std::size_t j = 0; j < N; ++j) {
            if (j == i) {
                continue;
            }
            next = {};
            for (std::size_t k = 0; k <= degree; ++k) {
                next[k] -= nodes[j] * coeffs[k];
                next[k + 1] += coeffs[k];
            }
            coeffs = next;
            ++degree;
        }

        Real denominator = Real(1);
        for (std::size_t j = 0; j < N; ++j) {
            if (j != i) {
                denominator *= nodes[i] - nodes[j];
            }
        }
        const Real inv_denominator = Real(1) / denominator;
        for (std::size_t k = 0; k < N; ++k) {
            table.values[i * N + k] = coeffs[k] * inv_denominator;
        }
        if constexpr (N >= 2) {
            for (std::size_t k = 1; k < N; ++k) {
                table.first[i * (N - 1) + (k - 1)] =
                    static_cast<Real>(k) * table.values[i * N + k];
            }
        }
        if constexpr (N >= 3) {
            for (std::size_t k = 2; k < N; ++k) {
                table.second[i * (N - 2) + (k - 2)] =
                    static_cast<Real>(k * (k - 1)) * table.values[i * N + k];
            }
        }
    }

    return table;
}

template<std::size_t N>
void assign_axis_coefficient_table(const AxisMonomialCoefficientTable<N>& table,
                                   std::vector<Real>& values,
                                   std::vector<Real>& first,
                                   std::vector<Real>& second) {
    assign_array(values, table.values);
    assign_array(first, table.first);
    assign_array(second, table.second);
}

bool assign_precomputed_axis_coefficients(int n_axis,
                                          std::vector<Real>& values,
                                          std::vector<Real>& first,
                                          std::vector<Real>& second) {
    static constexpr auto kAxisCoefficients1 = make_axis_monomial_coefficient_table<1>();
    static constexpr auto kAxisCoefficients2 = make_axis_monomial_coefficient_table<2>();
    static constexpr auto kAxisCoefficients3 = make_axis_monomial_coefficient_table<3>();
    static constexpr auto kAxisCoefficients4 = make_axis_monomial_coefficient_table<4>();
    static constexpr auto kAxisCoefficients5 = make_axis_monomial_coefficient_table<5>();

    switch (n_axis) {
        case 1:
            assign_axis_coefficient_table(kAxisCoefficients1, values, first, second);
            return true;
        case 2:
            assign_axis_coefficient_table(kAxisCoefficients2, values, first, second);
            return true;
        case 3:
            assign_axis_coefficient_table(kAxisCoefficients3, values, first, second);
            return true;
        case 4:
            assign_axis_coefficient_table(kAxisCoefficients4, values, first, second);
            return true;
        case 5:
            assign_axis_coefficient_table(kAxisCoefficients5, values, first, second);
            return true;
        default:
            return false;
    }
}

LagrangeTopologyTraits lagrange_topology_traits(ElementType type) {
    const auto topo = topology(type);
    if (topo != LagrangeTopology::Unknown) {
        return {topo, reference_dimension(type)};
    }

    throw BasisElementCompatibilityException("Unsupported element type for LagrangeBasis",
                                             __FILE__, __LINE__, __func__);
}

std::size_t lattice_index_pm_one(Real coord, int order, const char* context) {
    if (order <= 0) {
        if (!coordinate_matches_expected(coord, Real(0))) {
            throw BasisNodeOrderingException(context, __FILE__, __LINE__, __func__);
        }
        return 0;
    }

    const Real scaled = (coord + Real(1)) * static_cast<Real>(order) / Real(2);
    const long idx = std::lround(scaled);
    if (idx < 0 || idx > order ||
        !coordinate_matches_expected(
            coord,
            detail::equispaced_pm_one_coord(static_cast<int>(idx), order))) {
        throw BasisNodeOrderingException(context, __FILE__, __LINE__, __func__);
    }
    return static_cast<std::size_t>(idx);
}

int simplex_lattice_index(Real coord, int order, const char* context) {
    if (order <= 0) {
        if (!coordinate_matches_expected(coord, Real(0)) &&
            !coordinate_matches_expected(coord, Real(0.25)) &&
            !coordinate_matches_expected(coord, Real(1) / Real(3))) {
            throw BasisNodeOrderingException(context, __FILE__, __LINE__, __func__);
        }
        return 0;
    }

    const Real scaled = coord * static_cast<Real>(order);
    const long idx = std::lround(scaled);
    const Real reconstructed = static_cast<Real>(idx) / static_cast<Real>(order);
    if (idx < 0 || idx > order || !coordinate_matches_expected(coord, reconstructed)) {
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

// Horner-form evaluator for the precomputed 1D Lagrange basis.
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

void fill_equispaced_barycentric_weights(int n_axis, Real* weights) {
    const int order = n_axis - 1;
    Real weight = (order % 2 == 0) ? Real(1) : Real(-1);
    Real max_abs = Real(0);
    for (int i = 0; i < n_axis; ++i) {
        weights[i] = weight;
        max_abs = std::max(max_abs, std::abs(weight));
        if (i < order) {
            weight *= -static_cast<Real>(order - i) / static_cast<Real>(i + 1);
        }
    }

    if (max_abs > Real(0)) {
        const Real inv_scale = Real(1) / max_abs;
        for (int i = 0; i < n_axis; ++i) {
            weights[i] *= inv_scale;
        }
    }
}

bool coordinate_matches_axis_node(Real xi, Real node) {
    return coordinate_matches_expected(xi, node);
}

struct CompensatedSum {
    Real sum{Real(0)};
    Real compensation{Real(0)};

    void add(Real value) noexcept {
        const Real y = value - compensation;
        const Real t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
};

void distribute_residual_by_abs(int n_axis, Real* values, Real residual) {
    if (values == nullptr || n_axis <= 0 || residual == Real(0)) {
        return;
    }

    CompensatedSum abs_sum;
    int largest_index = 0;
    Real largest_abs = Real(0);
    for (int i = 0; i < n_axis; ++i) {
        const Real magnitude = std::abs(values[i]);
        abs_sum.add(magnitude);
        if (magnitude > largest_abs) {
            largest_abs = magnitude;
            largest_index = i;
        }
    }

    if (abs_sum.sum <= Real(0)) {
        values[0] += residual;
        return;
    }

    const Real inv_abs_sum = Real(1) / abs_sum.sum;
    CompensatedSum applied;
    for (int i = 0; i < n_axis; ++i) {
        const Real correction = residual * std::abs(values[i]) * inv_abs_sum;
        values[i] += correction;
        applied.add(correction);
    }
    values[largest_index] += residual - applied.sum;
}

void evaluate_1d_barycentric_runtime(int n_axis,
                                     Real xi,
                                     const Real* weights,
                                     Real* values,
                                     Real* first,
                                     Real* second) {
    const int order = n_axis - 1;
    BASIS_CHECK_EVAL(weights != nullptr,
                     "LagrangeBasis: missing cached barycentric weights for runtime axis evaluation");

    int node_index = -1;
    for (int i = 0; i < n_axis; ++i) {
        const Real node = detail::equispaced_pm_one_coord(i, order);
        if (coordinate_matches_axis_node(xi, node)) {
            node_index = i;
            break;
        }
    }

    if (node_index >= 0) {
        std::fill(values, values + n_axis, Real(0));
        values[node_index] = Real(1);
        if (!first && !second) {
            return;
        }

        const Real xk = detail::equispaced_pm_one_coord(node_index, order);
        const Real wk = weights[static_cast<std::size_t>(node_index)];
        Real reciprocal_sum = Real(0);
        if (second) {
            for (int m = 0; m < n_axis; ++m) {
                if (m == node_index) {
                    continue;
                }
                const Real xm = detail::equispaced_pm_one_coord(m, order);
                reciprocal_sum += Real(1) / (xk - xm);
            }
        }

        Real first_diagonal = Real(0);
        Real second_diagonal = Real(0);
        if (first) {
            std::fill(first, first + n_axis, Real(0));
        }
        if (second) {
            std::fill(second, second + n_axis, Real(0));
        }

        for (int j = 0; j < n_axis; ++j) {
            if (j == node_index) {
                continue;
            }
            const Real xj = detail::equispaced_pm_one_coord(j, order);
            const Real distance = xk - xj;
            const Real offdiag_first = weights[static_cast<std::size_t>(j)] / (wk * distance);
            first_diagonal -= offdiag_first;
            if (first) {
                first[j] = offdiag_first;
            }
            if (second) {
                const Real offdiag_second =
                    Real(2) * offdiag_first * (reciprocal_sum - Real(1) / distance);
                second[j] = offdiag_second;
                second_diagonal -= offdiag_second;
            }
        }
        if (first) {
            first[node_index] = first_diagonal;
        }
        if (second) {
            second[node_index] = second_diagonal;
        }
        return;
    }

    Real sum0 = Real(0);
    Real sum1 = Real(0);
    Real sum2 = Real(0);
    for (int i = 0; i < n_axis; ++i) {
        const Real node = detail::equispaced_pm_one_coord(i, order);
        const Real inv_distance = Real(1) / (xi - node);
        const Real weighted = weights[static_cast<std::size_t>(i)] * inv_distance;
        sum0 += weighted;
        sum1 += weighted * inv_distance;
        sum2 += weighted * inv_distance * inv_distance;
    }

    const Real inv_sum0 = Real(1) / sum0;
    const Real first_ratio = sum1 * inv_sum0;
    const Real second_ratio = sum2 * inv_sum0;
    const Real first_ratio_sq = first_ratio * first_ratio;

    CompensatedSum value_sum;
    CompensatedSum first_sum;
    CompensatedSum second_sum;
    for (int i = 0; i < n_axis; ++i) {
        const Real node = detail::equispaced_pm_one_coord(i, order);
        const Real inv_distance = Real(1) / (xi - node);
        const Real value = weights[static_cast<std::size_t>(i)] * inv_distance * inv_sum0;
        values[i] = value;
        value_sum.add(value);
        if (first || second) {
            const Real derivative_factor = first_ratio - inv_distance;
            if (first) {
                first[i] = value * derivative_factor;
                first_sum.add(first[i]);
            }
            if (second) {
                second[i] = value * (derivative_factor * derivative_factor +
                                     inv_distance * inv_distance -
                                     Real(2) * second_ratio +
                                     first_ratio_sq);
                second_sum.add(second[i]);
            }
        }
    }

    distribute_residual_by_abs(n_axis, values, Real(1) - value_sum.sum);
    if (first) {
        distribute_residual_by_abs(n_axis, first, -first_sum.sum);
    }
    if (second) {
        distribute_residual_by_abs(n_axis, second, -second_sum.sum);
    }
}

// 1D Lagrange-basis evaluator. Writes n_axis entries to each non-null output
// buffer. Dispatches to compile-time Horner specializations for sizes 1..5
// (orders 0..4 — the common cases) and uses barycentric evaluation above that
// threshold to avoid high-order monomial conditioning issues.
void evaluate_1d_basis_to(const Real* v_coeffs,
                          const Real* d_coeffs,
                          const Real* d2_coeffs,
                          const Real* barycentric_weights,
                          int n_axis, Real xi,
                          Real* values, Real* first, Real* second) {
    switch (n_axis) {
        case 1: evaluate_1d_horner_impl<1>(v_coeffs, d_coeffs, d2_coeffs, xi, values, first, second); return;
        case 2: evaluate_1d_horner_impl<2>(v_coeffs, d_coeffs, d2_coeffs, xi, values, first, second); return;
        case 3: evaluate_1d_horner_impl<3>(v_coeffs, d_coeffs, d2_coeffs, xi, values, first, second); return;
        case 4: evaluate_1d_horner_impl<4>(v_coeffs, d_coeffs, d2_coeffs, xi, values, first, second); return;
        case 5: evaluate_1d_horner_impl<5>(v_coeffs, d_coeffs, d2_coeffs, xi, values, first, second); return;
        default:
            evaluate_1d_barycentric_runtime(n_axis, xi, barycentric_weights, values, first, second);
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
    std::vector<Real> tri_gradient_components;
    std::vector<Real> tri_hessian_components;
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
// only read the slots they requested via `level`. Common low orders use
// precomputed Horner coefficients; high orders use barycentric axis evaluation.
AxisBasisEvaluations fill_axis_scratch(AxisScratch& s,
                                       const Real* v_coeffs,
                                       const Real* d_coeffs,
                                       const Real* d2_coeffs,
                                       const Real* barycentric_weights,
                                       int n_axis, Real xi,
                                       AxisDeriv level) {
    const std::size_t n = static_cast<std::size_t>(n_axis);
    s.reserveFor(n);
    Real* first  = (level == AxisDeriv::ValuesOnly) ? nullptr : s.first.data();
    Real* second = (level == AxisDeriv::ValuesAndFirstAndSecond) ? s.second.data() : nullptr;
    evaluate_1d_basis_to(v_coeffs, d_coeffs, d2_coeffs, barycentric_weights,
                         n_axis, xi, s.values.data(), first, second);
    return AxisBasisEvaluations{s.values.data(), s.first.data(), s.second.data(), n};
}

struct AxisBatchScratch {
    std::vector<Real> values;
    std::vector<Real> first;
    std::vector<Real> second;

    void resizeFor(std::size_t count, AxisDeriv level) {
        if (values.size() < count) values.resize(count);
        if (level != AxisDeriv::ValuesOnly && first.size() < count) first.resize(count);
        if (level == AxisDeriv::ValuesAndFirstAndSecond && second.size() < count) second.resize(count);
    }
};

void fill_axis_batch(AxisBatchScratch& scratch,
                     const std::vector<math::Vector<Real, 3>>& points,
                     std::size_t component,
                     const Real* v_coeffs,
                     const Real* d_coeffs,
                     const Real* d2_coeffs,
                     const Real* barycentric_weights,
                     int n_axis,
                     AxisDeriv level) {
    const std::size_t count = points.size() * static_cast<std::size_t>(n_axis);
    scratch.resizeFor(count, level);
    Real* first = (level == AxisDeriv::ValuesOnly) ? nullptr : scratch.first.data();
    Real* second = (level == AxisDeriv::ValuesAndFirstAndSecond) ? scratch.second.data() : nullptr;
    const std::size_t axis_stride = static_cast<std::size_t>(n_axis);
    for (std::size_t q = 0; q < points.size(); ++q) {
        evaluate_1d_basis_to(v_coeffs, d_coeffs, d2_coeffs, barycentric_weights, n_axis,
                             points[q][component],
                             scratch.values.data() + q * axis_stride,
                             first ? first + q * axis_stride : nullptr,
                             second ? second + q * axis_stride : nullptr);
    }
}

// Maximum yz-table footprint that fits comfortably on the stack for typical
// element orders. Up to order 7 per axis (8x8 yz pairs = 64 entries per table).
// Higher orders fall back to thread_local heap buffers.
inline constexpr std::size_t kMaxStackYZ = 64;

struct TensorProductVectorSink {
    std::vector<Real>* values;
    std::vector<Gradient>* gradients;
    std::vector<Hessian>* hessians;

    bool wants_values() const noexcept { return values != nullptr; }
    bool wants_gradients() const noexcept { return gradients != nullptr; }
    bool wants_hessians() const noexcept { return hessians != nullptr; }

    void prepare(std::size_t n_nodes) const {
        if (values)    values->resize(n_nodes);
        if (gradients) gradients->resize(n_nodes);
        if (hessians)  hessians->resize(n_nodes);
    }

    void write_value(std::size_t n, Real value) const {
        (*values)[n] = value;
    }

    void write_gradient(std::size_t n, Real dx, Real dy, Real dz) const {
        auto& g = (*gradients)[n];
        g[0] = dx;
        g[1] = dy;
        g[2] = dz;
    }

    void write_hessian(std::size_t n,
                       Real xx,
                       Real yy,
                       Real zz,
                       Real xy,
                       Real xz,
                       Real yz) const {
        (*hessians)[n] = make_symmetric_hessian(xx, yy, zz, xy, xz, yz);
    }
};

struct TensorProductRawSink {
    Real* values;
    Real* gradients;
    Real* hessians;

    bool wants_values() const noexcept { return values != nullptr; }
    bool wants_gradients() const noexcept { return gradients != nullptr; }
    bool wants_hessians() const noexcept { return hessians != nullptr; }

    void prepare(std::size_t) const {}

    void write_value(std::size_t n, Real value) const {
        values[n] = value;
    }

    void write_gradient(std::size_t n, Real dx, Real dy, Real dz) const {
        Real* g = gradients + n * 3u;
        g[0] = dx;
        g[1] = dy;
        g[2] = dz;
    }

    void write_hessian(std::size_t n,
                       Real xx,
                       Real yy,
                       Real zz,
                       Real xy,
                       Real xz,
                       Real yz) const {
        Real* H = hessians + n * 9u;
        H[0] = xx;
        H[4] = yy;
        H[8] = zz;
        H[1] = xy; H[3] = xy;
        H[2] = xz; H[6] = xz;
        H[5] = yz; H[7] = yz;
    }
};

// Fused sum-factorized tensor-product evaluator.
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
//   all three         : 10 (was 20)
//
// Dimensional scope: works uniformly for Line/Quadrilateral/Hexahedron with
// the unused axes' size folded to 1 via constant_axis_basis().
template <typename Sink>
void evaluate_tensor_product_factorized_impl(
    const std::vector<std::array<std::size_t, 3>>& tensor_indices,
    const AxisBasisEvaluations& x_axis,
    const AxisBasisEvaluations& y_axis,
    const AxisBasisEvaluations& z_axis,
    const Sink& sink) {
    const std::size_t ny = y_axis.size;
    const std::size_t nz = z_axis.size;
    const std::size_t nyz = ny * nz;
    const bool need_values = sink.wants_values();
    const bool need_grad = sink.wants_gradients();
    const bool need_hess = sink.wants_hessians();

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
    sink.prepare(n_nodes);

    for (std::size_t n = 0; n < n_nodes; ++n) {
        const auto& idx = tensor_indices[n];
        const std::size_t i = idx[0];
        const std::size_t jk = idx[1] * nz + idx[2];

        const Real Lx = x_axis.values[i];

        if (need_values) {
            sink.write_value(n, Lx * Mvv[jk]);
        }

        if (need_grad) {
            const Real dLx = x_axis.first[i];
            sink.write_gradient(n,
                                dLx * Mvv[jk],
                                Lx  * Mdv[jk],
                                Lx  * Mvd[jk]);
        }

        if (need_hess) {
            const Real dLx  = x_axis.first[i];
            const Real d2Lx = x_axis.second[i];
            sink.write_hessian(n,
                               d2Lx * Mvv[jk],
                               Lx   * Md2v[jk],
                               Lx   * Mvd2[jk],
                               dLx  * Mdv[jk],
                               dLx  * Mvd[jk],
                               Lx   * Mdd[jk]);
        }
    }
}

void evaluate_tensor_product_factorized(
    const std::vector<std::array<std::size_t, 3>>& tensor_indices,
    const AxisBasisEvaluations& x_axis,
    const AxisBasisEvaluations& y_axis,
    const AxisBasisEvaluations& z_axis,
    std::vector<Real>* values_out,
    std::vector<Gradient>* gradients_out,
    std::vector<Hessian>* hessians_out) {
    const TensorProductVectorSink sink{values_out, gradients_out, hessians_out};
    evaluate_tensor_product_factorized_impl(tensor_indices, x_axis, y_axis, z_axis, sink);
}

void evaluate_tensor_product_factorized_to(
    const std::vector<std::array<std::size_t, 3>>& tensor_indices,
    const AxisBasisEvaluations& x_axis,
    const AxisBasisEvaluations& y_axis,
    const AxisBasisEvaluations& z_axis,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) {
    const TensorProductRawSink sink{values_out, gradients_out, hessians_out};
    evaluate_tensor_product_factorized_impl(tensor_indices, x_axis, y_axis, z_axis, sink);
}

void evaluate_tensor_product_points_strided(
    LagrangeTopology topology,
    const std::vector<std::array<std::size_t, 3>>& tensor_indices,
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    const Real* v_coeffs,
    const Real* d_coeffs,
    const Real* d2_coeffs,
    const Real* barycentric_weights,
    int n_axis,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) {
    const std::size_t num_qpts = points.size();
    if (num_qpts == 0 || tensor_indices.empty()) {
        return;
    }

    const bool need_grad = gradients_out != nullptr;
    const bool need_hess = hessians_out != nullptr;
    const AxisDeriv level = need_hess
        ? AxisDeriv::ValuesAndFirstAndSecond
        : (need_grad ? AxisDeriv::ValuesAndFirst : AxisDeriv::ValuesOnly);

    thread_local AxisBatchScratch x_batch;
    thread_local AxisBatchScratch y_batch;
    thread_local AxisBatchScratch z_batch;

    fill_axis_batch(x_batch, points, 0u, v_coeffs, d_coeffs, d2_coeffs,
                    barycentric_weights, n_axis, level);
    const bool has_y = topology != LagrangeTopology::Line;
    const bool has_z = topology == LagrangeTopology::Hexahedron;
    if (has_y) {
        fill_axis_batch(y_batch, points, 1u, v_coeffs, d_coeffs, d2_coeffs,
                        barycentric_weights, n_axis, level);
    }
    if (has_z) {
        fill_axis_batch(z_batch, points, 2u, v_coeffs, d_coeffs, d2_coeffs,
                        barycentric_weights, n_axis, level);
    }

    const std::size_t axis_stride = static_cast<std::size_t>(n_axis);
    for (std::size_t node = 0; node < tensor_indices.size(); ++node) {
        const auto& idx = tensor_indices[node];
        const std::size_t i = idx[0];
        const std::size_t j = idx[1];
        const std::size_t k = idx[2];

        Real* value_row = values_out ? values_out + node * output_stride : nullptr;
        Real* grad_row = gradients_out ? gradients_out + node * 3u * output_stride : nullptr;
        Real* hess_row = hessians_out ? hessians_out + node * 9u * output_stride : nullptr;

        for (std::size_t q = 0; q < num_qpts; ++q) {
            const std::size_t q_axis = q * axis_stride;
            const Real xv = x_batch.values[q_axis + i];
            const Real yv = has_y ? y_batch.values[q_axis + j] : Real(1);
            const Real zv = has_z ? z_batch.values[q_axis + k] : Real(1);

            if (value_row != nullptr) {
                value_row[q] = xv * yv * zv;
            }

            if (need_grad) {
                const Real xd = x_batch.first[q_axis + i];
                const Real yd = has_y ? y_batch.first[q_axis + j] : Real(0);
                const Real zd = has_z ? z_batch.first[q_axis + k] : Real(0);
                grad_row[0u * output_stride + q] = xd * yv * zv;
                grad_row[1u * output_stride + q] = xv * yd * zv;
                grad_row[2u * output_stride + q] = xv * yv * zd;
            }

            if (need_hess) {
                const Real xd = x_batch.first[q_axis + i];
                const Real yd = has_y ? y_batch.first[q_axis + j] : Real(0);
                const Real zd = has_z ? z_batch.first[q_axis + k] : Real(0);
                const Real x2 = x_batch.second[q_axis + i];
                const Real y2 = has_y ? y_batch.second[q_axis + j] : Real(0);
                const Real z2 = has_z ? z_batch.second[q_axis + k] : Real(0);

                hess_row[0u * output_stride + q] = x2 * yv * zv;
                hess_row[4u * output_stride + q] = xv * y2 * zv;
                hess_row[8u * output_stride + q] = xv * yv * z2;
                hess_row[1u * output_stride + q] = xd * yd * zv;
                hess_row[3u * output_stride + q] = hess_row[1u * output_stride + q];
                hess_row[2u * output_stride + q] = xd * yv * zd;
                hess_row[6u * output_stride + q] = hess_row[2u * output_stride + q];
                hess_row[5u * output_stride + q] = xv * yd * zd;
                hess_row[7u * output_stride + q] = hess_row[5u * output_stride + q];
            }
        }
    }
}

void evaluate_wedge_points_strided(
    const std::vector<std::array<int, 4>>& simplex_exponents,
    const std::vector<std::array<std::size_t, 2>>& wedge_indices,
    int order,
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    const Real* v_coeffs,
    const Real* d_coeffs,
    const Real* d2_coeffs,
    const Real* barycentric_weights,
    int n_axis,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) {
    if (points.empty() || wedge_indices.empty()) {
        return;
    }

    const bool need_grad = gradients_out != nullptr;
    const bool need_hess = hessians_out != nullptr;
    const AxisDeriv level = need_hess
        ? AxisDeriv::ValuesAndFirstAndSecond
        : (need_grad ? AxisDeriv::ValuesAndFirst : AxisDeriv::ValuesOnly);

    LagrangeEvaluateScratch& scratch = evaluate_scratch();
    const std::size_t tri_count = simplex_exponents.size();
    scratch.tri_values.resize(tri_count);
    if (need_grad || need_hess) {
        scratch.tri_gradient_components.resize(tri_count * 3u);
    }
    if (need_hess) {
        scratch.tri_hessian_components.resize(tri_count * 9u);
    }

    for (std::size_t q = 0; q < points.size(); ++q) {
        const auto& xi = points[q];
        const AxisBasisEvaluations z_axis =
            fill_axis_scratch(scratch.axis_z, v_coeffs, d_coeffs, d2_coeffs,
                              barycentric_weights, n_axis, xi[2], level);

        detail::evaluate_triangle_simplex_basis_to(
            simplex_exponents,
            order,
            xi,
            scratch.tri_values.data(),
            (need_grad || need_hess) ? scratch.tri_gradient_components.data() : nullptr,
            need_hess ? scratch.tri_hessian_components.data() : nullptr);

        for (std::size_t node = 0; node < wedge_indices.size(); ++node) {
            const auto& index = wedge_indices[node];
            const std::size_t tri = index[0];
            const std::size_t z = index[1];
            const Real tri_v = scratch.tri_values[tri];
            const Real zv = z_axis.values[z];

            if (values_out != nullptr) {
                values_out[node * output_stride + q] = tri_v * zv;
            }

            if (need_grad) {
                const Real* tri_g = scratch.tri_gradient_components.data() + tri * 3u;
                Real* g = gradients_out + node * 3u * output_stride;
                g[0u * output_stride + q] = tri_g[0] * zv;
                g[1u * output_stride + q] = tri_g[1] * zv;
                g[2u * output_stride + q] = tri_v * z_axis.first[z];
            }

            if (need_hess) {
                const Real* tri_g = scratch.tri_gradient_components.data() + tri * 3u;
                const Real* tri_H = scratch.tri_hessian_components.data() + tri * 9u;
                const Real zd = z_axis.first[z];
                Real* H = hessians_out + node * 9u * output_stride;
                H[0u * output_stride + q] = tri_H[0] * zv;
                H[1u * output_stride + q] = tri_H[1] * zv;
                H[2u * output_stride + q] = tri_g[0] * zd;
                H[3u * output_stride + q] = H[1u * output_stride + q];
                H[4u * output_stride + q] = tri_H[4] * zv;
                H[5u * output_stride + q] = tri_g[1] * zd;
                H[6u * output_stride + q] = H[2u * output_stride + q];
                H[7u * output_stride + q] = H[5u * output_stride + q];
                H[8u * output_stride + q] = tri_v * z_axis.second[z];
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
    init_evaluation_dispatch();
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
    axis_barycentric_weights_.clear();
    const auto topology = lagrange_topology_traits(element_type_).topology;
    topology_id_ = static_cast<int>(topology);
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
        case LagrangeTopology::Unknown:
            break;
    }

    throw BasisElementCompatibilityException("Unsupported element type in LagrangeBasis::init_nodes",
                                             __FILE__, __LINE__, __func__);
}

void LagrangeBasis::init_evaluation_dispatch() {
    const auto topology = static_cast<LagrangeTopology>(topology_id_);
    switch (topology) {
        case LagrangeTopology::Point:
            vector_evaluation_dispatch_ = &LagrangeBasis::evaluate_point_vectors;
            return;
        case LagrangeTopology::Line:
        case LagrangeTopology::Quadrilateral:
        case LagrangeTopology::Hexahedron:
            vector_evaluation_dispatch_ = &LagrangeBasis::evaluate_tensor_product_vectors;
            return;
        case LagrangeTopology::Triangle:
            vector_evaluation_dispatch_ = &LagrangeBasis::evaluate_triangle_vectors;
            return;
        case LagrangeTopology::Tetrahedron:
            vector_evaluation_dispatch_ = &LagrangeBasis::evaluate_tetrahedron_vectors;
            return;
        case LagrangeTopology::Wedge:
            vector_evaluation_dispatch_ = &LagrangeBasis::evaluate_wedge_vectors;
            return;
        case LagrangeTopology::Pyramid:
            vector_evaluation_dispatch_ = &LagrangeBasis::evaluate_pyramid_vectors;
            return;
        case LagrangeTopology::Unknown:
            break;
    }
    vector_evaluation_dispatch_ = &LagrangeBasis::evaluate_unsupported_vectors;
}

void LagrangeBasis::compute_axis_monomial_coefficients() {
    const int N = static_cast<int>(nodes_1d_.size());
    if (N == 0) return;

    axis_barycentric_weights_.resize(static_cast<std::size_t>(N));
    fill_equispaced_barycentric_weights(N, axis_barycentric_weights_.data());

    if (assign_precomputed_axis_coefficients(N, axis_v_coeffs_, axis_d_coeffs_, axis_d2_coeffs_)) {
        return;
    }

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

    nodes_ = ReferenceNodeLayout::get_lagrange_node_coords(element_type_, order_);
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
    nodes_ = ReferenceNodeLayout::get_lagrange_node_coords(element_type_, order_);
    const auto topology = static_cast<LagrangeTopology>(topology_id_);
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
    const auto triangle_nodes = ReferenceNodeLayout::get_lagrange_node_coords(ElementType::Triangle3, order_);
    simplex_exponents_.clear();
    simplex_exponents_.reserve(triangle_nodes.size());
    std::unordered_map<std::array<int, 4>, std::size_t, SimplexExponentHash> triangle_index_by_exponent;
    triangle_index_by_exponent.reserve(triangle_nodes.size());
    for (std::size_t tri = 0; tri < triangle_nodes.size(); ++tri) {
        const auto exponents = triangle_exponents_from_public_node(triangle_nodes[tri], order_);
        simplex_exponents_.push_back(exponents);
        const auto inserted = triangle_index_by_exponent.emplace(exponents, tri);
        if (!inserted.second) {
            throw BasisNodeOrderingException("LagrangeBasis: duplicate wedge triangle descriptor",
                                             __FILE__, __LINE__, __func__);
        }
    }

    nodes_ = ReferenceNodeLayout::get_lagrange_node_coords(element_type_, order_);
    wedge_indices_.clear();
    wedge_indices_.reserve(nodes_.size());
    for (const auto& node : nodes_) {
        const auto exponents = triangle_exponents_from_public_node(node, order_);
        const auto found = triangle_index_by_exponent.find(exponents);
        if (found == triangle_index_by_exponent.end()) {
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
    nodes_ = detail::lagrange_pyramid::nodes(order_);
}

void LagrangeBasis::evaluate_point_vectors(const math::Vector<Real, 3>&,
                                           std::vector<Real>* values,
                                           std::vector<Gradient>* gradients,
                                           std::vector<Hessian>* hessians) const {
    if (values != nullptr) {
        values->resize(1u);
        (*values)[0] = Real(1);
    }
    if (gradients != nullptr) {
        gradients->resize(1u);
        (*gradients)[0] = Gradient{};
    }
    if (hessians != nullptr) {
        hessians->resize(1u);
        (*hessians)[0] = Hessian{};
    }
}

void LagrangeBasis::evaluate_tensor_product_vectors(const math::Vector<Real, 3>& xi,
                                                    std::vector<Real>* values,
                                                    std::vector<Gradient>* gradients,
                                                    std::vector<Hessian>* hessians) const {
    const auto topology = static_cast<LagrangeTopology>(topology_id_);
    if (evaluate_fixed_lagrange_fast(topology, order_, xi, values, gradients, hessians)) {
        return;
    }

    const int n_axis = static_cast<int>(nodes_1d_.size());
    const Real* vc = axis_v_coeffs_.data();
    const Real* dc = axis_d_coeffs_.data();
    const Real* d2c = axis_d2_coeffs_.data();
    const Real* bw = axis_barycentric_weights_.data();
    const AxisDeriv level = hessians != nullptr ? AxisDeriv::ValuesAndFirstAndSecond
                           : gradients != nullptr ? AxisDeriv::ValuesAndFirst
                                                  : AxisDeriv::ValuesOnly;

    LagrangeEvaluateScratch& scratch = evaluate_scratch();
    const AxisBasisEvaluations x_axis =
        fill_axis_scratch(scratch.axis_x, vc, dc, d2c, bw, n_axis, xi[0], level);
    AxisBasisEvaluations y_axis = constant_axis_basis();
    AxisBasisEvaluations z_axis = constant_axis_basis();

    if (topology != LagrangeTopology::Line) {
        y_axis = fill_axis_scratch(scratch.axis_y, vc, dc, d2c, bw, n_axis, xi[1], level);
    }
    if (topology == LagrangeTopology::Hexahedron) {
        z_axis = fill_axis_scratch(scratch.axis_z, vc, dc, d2c, bw, n_axis, xi[2], level);
    }

    evaluate_tensor_product_factorized(tensor_indices_, x_axis, y_axis, z_axis,
                                       values, gradients, hessians);
}

void LagrangeBasis::evaluate_triangle_vectors(const math::Vector<Real, 3>& xi,
                                              std::vector<Real>* values,
                                              std::vector<Gradient>* gradients,
                                              std::vector<Hessian>* hessians) const {
    const auto topology = static_cast<LagrangeTopology>(topology_id_);
    if (evaluate_fixed_lagrange_fast(topology, order_, xi, values, gradients, hessians)) {
        return;
    }
    detail::evaluate_triangle_simplex_basis(simplex_exponents_, order_, xi,
                                            values, gradients, hessians);
}

void LagrangeBasis::evaluate_tetrahedron_vectors(const math::Vector<Real, 3>& xi,
                                                 std::vector<Real>* values,
                                                 std::vector<Gradient>* gradients,
                                                 std::vector<Hessian>* hessians) const {
    const auto topology = static_cast<LagrangeTopology>(topology_id_);
    if (evaluate_fixed_lagrange_fast(topology, order_, xi, values, gradients, hessians)) {
        return;
    }
    detail::evaluate_tetrahedron_simplex_basis(simplex_exponents_, order_, xi,
                                               values, gradients, hessians);
}

void LagrangeBasis::evaluate_wedge_vectors(const math::Vector<Real, 3>& xi,
                                           std::vector<Real>* values,
                                           std::vector<Gradient>* gradients,
                                           std::vector<Hessian>* hessians) const {
    const int n_axis = static_cast<int>(nodes_1d_.size());
    const Real* vc = axis_v_coeffs_.data();
    const Real* dc = axis_d_coeffs_.data();
    const Real* d2c = axis_d2_coeffs_.data();
    const Real* bw = axis_barycentric_weights_.data();
    const AxisDeriv level = hessians != nullptr ? AxisDeriv::ValuesAndFirstAndSecond
                           : gradients != nullptr ? AxisDeriv::ValuesAndFirst
                                                  : AxisDeriv::ValuesOnly;

    LagrangeEvaluateScratch& scratch = evaluate_scratch();
    const AxisBasisEvaluations z_axis =
        fill_axis_scratch(scratch.axis_z, vc, dc, d2c, bw, n_axis, xi[2], level);

    if (hessians != nullptr) {
        detail::evaluate_triangle_simplex_basis(
            simplex_exponents_, order_, xi,
            &scratch.tri_values, &scratch.tri_gradients, &scratch.tri_hessians);
    } else if (gradients != nullptr) {
        detail::evaluate_triangle_simplex_basis(
            simplex_exponents_, order_, xi,
            &scratch.tri_values, &scratch.tri_gradients, nullptr);
    } else {
        detail::evaluate_triangle_simplex_basis(
            simplex_exponents_, order_, xi,
            &scratch.tri_values, nullptr, nullptr);
    }

    const std::size_t n_nodes = wedge_indices_.size();
    if (values != nullptr) {
        values->resize(n_nodes);
    }
    if (gradients != nullptr) {
        gradients->resize(n_nodes);
    }
    if (hessians != nullptr) {
        hessians->resize(n_nodes);
    }

    for (std::size_t n = 0; n < n_nodes; ++n) {
        const auto& index = wedge_indices_[n];
        const std::size_t tri_idx = index[0];
        const std::size_t z_idx = index[1];
        const Real zv = z_axis.values[z_idx];
        const Real tri_v = scratch.tri_values[tri_idx];

        if (values != nullptr) {
            (*values)[n] = tri_v * zv;
        }
        if (gradients != nullptr) {
            const Real zd = z_axis.first[z_idx];
            (*gradients)[n][0] = scratch.tri_gradients[tri_idx][0] * zv;
            (*gradients)[n][1] = scratch.tri_gradients[tri_idx][1] * zv;
            (*gradients)[n][2] = tri_v * zd;
        }
        if (hessians != nullptr) {
            const Real zd = z_axis.first[z_idx];
            const Real zd2 = z_axis.second[z_idx];
            Hessian H{};
            H(0, 0) = scratch.tri_hessians[tri_idx](0, 0) * zv;
            H(1, 1) = scratch.tri_hessians[tri_idx](1, 1) * zv;
            H(0, 1) = scratch.tri_hessians[tri_idx](0, 1) * zv;
            H(1, 0) = H(0, 1);
            H(2, 2) = tri_v * zd2;
            H(0, 2) = scratch.tri_gradients[tri_idx][0] * zd;
            H(2, 0) = H(0, 2);
            H(1, 2) = scratch.tri_gradients[tri_idx][1] * zd;
            H(2, 1) = H(1, 2);
            (*hessians)[n] = H;
        }
    }
}

void LagrangeBasis::evaluate_pyramid_vectors(const math::Vector<Real, 3>& xi,
                                             std::vector<Real>* values,
                                             std::vector<Gradient>* gradients,
                                             std::vector<Hessian>* hessians) const {
    if (values != nullptr && gradients != nullptr && hessians != nullptr) {
        detail::lagrange_pyramid::evaluate_all(order_, xi, *values, *gradients, *hessians);
        return;
    }
    if (values != nullptr) {
        detail::lagrange_pyramid::evaluate_values(order_, xi, *values);
    }
    if (gradients != nullptr) {
        detail::lagrange_pyramid::evaluate_gradients(order_, xi, *gradients);
    }
    if (hessians != nullptr) {
        detail::lagrange_pyramid::evaluate_hessians(order_, xi, *hessians);
    }
}

void LagrangeBasis::evaluate_unsupported_vectors(const math::Vector<Real, 3>&,
                                                 std::vector<Real>*,
                                                 std::vector<Gradient>*,
                                                 std::vector<Hessian>*) const {
    throw BasisEvaluationException("Unsupported element in LagrangeBasis vector evaluation",
                                   __FILE__, __LINE__, __func__);
}

void LagrangeBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                    std::vector<Real>& values) const {
    (this->*vector_evaluation_dispatch_)(xi, &values, nullptr, nullptr);
}

void LagrangeBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                       std::vector<Gradient>& gradients) const {
    (this->*vector_evaluation_dispatch_)(xi, nullptr, &gradients, nullptr);
}

void LagrangeBasis::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                      std::vector<Hessian>& hessians) const {
    (this->*vector_evaluation_dispatch_)(xi, nullptr, nullptr, &hessians);
}

void LagrangeBasis::evaluate_all(const math::Vector<Real, 3>& xi,
                                 std::vector<Real>& values,
                                 std::vector<Gradient>& gradients,
                                 std::vector<Hessian>& hessians) const {
    (this->*vector_evaluation_dispatch_)(xi, &values, &gradients, &hessians);
}

void LagrangeBasis::evaluate_values_to(const math::Vector<Real, 3>& xi,
                                       Real* SVMP_RESTRICT values_out) const {
    const auto topology = static_cast<LagrangeTopology>(topology_id_);
    if (evaluate_fixed_lagrange_fast_to(topology, order_, xi, values_out, nullptr, nullptr)) {
        return;
    }

    const int n_axis = static_cast<int>(nodes_1d_.size());
    const Real* vc = axis_v_coeffs_.data();
    const Real* dc = axis_d_coeffs_.data();
    const Real* d2c = axis_d2_coeffs_.data();
    const Real* bw = axis_barycentric_weights_.data();
    switch (topology) {
        case LagrangeTopology::Point:
            values_out[0] = Real(1);
            return;
        case LagrangeTopology::Line:
        case LagrangeTopology::Quadrilateral:
        case LagrangeTopology::Hexahedron: {
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations x_axis =
                fill_axis_scratch(scratch.axis_x, vc, dc, d2c, bw, n_axis, xi[0], AxisDeriv::ValuesOnly);
            AxisBasisEvaluations y_axis = constant_axis_basis();
            AxisBasisEvaluations z_axis = constant_axis_basis();
            if (topology != LagrangeTopology::Line) {
                y_axis = fill_axis_scratch(scratch.axis_y, vc, dc, d2c, bw, n_axis, xi[1], AxisDeriv::ValuesOnly);
            }
            if (topology == LagrangeTopology::Hexahedron) {
                z_axis = fill_axis_scratch(scratch.axis_z, vc, dc, d2c, bw, n_axis, xi[2], AxisDeriv::ValuesOnly);
            }
            evaluate_tensor_product_factorized_to(tensor_indices_, x_axis, y_axis, z_axis,
                                                  values_out, nullptr, nullptr);
            return;
        }
        case LagrangeTopology::Triangle:
            detail::evaluate_triangle_simplex_basis_to(simplex_exponents_, order_, xi,
                                                       values_out, nullptr, nullptr);
            return;
        case LagrangeTopology::Tetrahedron:
            detail::evaluate_tetrahedron_simplex_basis_to(simplex_exponents_, order_, xi,
                                                          values_out, nullptr, nullptr);
            return;
        case LagrangeTopology::Wedge: {
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations z_axis =
                fill_axis_scratch(scratch.axis_z, vc, dc, d2c, bw, n_axis, xi[2], AxisDeriv::ValuesOnly);
            scratch.tri_values.resize(simplex_exponents_.size());
            detail::evaluate_triangle_simplex_basis_to(simplex_exponents_, order_, xi,
                                                       scratch.tri_values.data(), nullptr, nullptr);
            for (std::size_t n = 0; n < wedge_indices_.size(); ++n) {
                const auto& index = wedge_indices_[n];
                values_out[n] = scratch.tri_values[index[0]] * z_axis.values[index[1]];
            }
            return;
        }
        case LagrangeTopology::Pyramid: {
            detail::lagrange_pyramid::evaluate_values_to(order_, xi, values_out);
            return;
        }
        case LagrangeTopology::Unknown:
            break;
    }

    throw BasisEvaluationException("Unsupported element in evaluate_values_to",
                                   __FILE__, __LINE__, __func__);
}

void LagrangeBasis::evaluate_gradients_to(const math::Vector<Real, 3>& xi,
                                          Real* SVMP_RESTRICT gradients_out) const {
    const auto topology = static_cast<LagrangeTopology>(topology_id_);
    if (evaluate_fixed_lagrange_fast_to(topology, order_, xi, nullptr, gradients_out, nullptr)) {
        return;
    }

    const int n_axis = static_cast<int>(nodes_1d_.size());
    const Real* vc = axis_v_coeffs_.data();
    const Real* dc = axis_d_coeffs_.data();
    const Real* d2c = axis_d2_coeffs_.data();
    const Real* bw = axis_barycentric_weights_.data();
    switch (topology) {
        case LagrangeTopology::Point:
            gradients_out[0] = Real(0);
            gradients_out[1] = Real(0);
            gradients_out[2] = Real(0);
            return;
        case LagrangeTopology::Line:
        case LagrangeTopology::Quadrilateral:
        case LagrangeTopology::Hexahedron: {
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations x_axis =
                fill_axis_scratch(scratch.axis_x, vc, dc, d2c, bw, n_axis, xi[0], AxisDeriv::ValuesAndFirst);
            AxisBasisEvaluations y_axis = constant_axis_basis();
            AxisBasisEvaluations z_axis = constant_axis_basis();
            if (topology != LagrangeTopology::Line) {
                y_axis = fill_axis_scratch(scratch.axis_y, vc, dc, d2c, bw, n_axis, xi[1], AxisDeriv::ValuesAndFirst);
            }
            if (topology == LagrangeTopology::Hexahedron) {
                z_axis = fill_axis_scratch(scratch.axis_z, vc, dc, d2c, bw, n_axis, xi[2], AxisDeriv::ValuesAndFirst);
            }
            evaluate_tensor_product_factorized_to(tensor_indices_, x_axis, y_axis, z_axis,
                                                  nullptr, gradients_out, nullptr);
            return;
        }
        case LagrangeTopology::Triangle:
            detail::evaluate_triangle_simplex_basis_to(simplex_exponents_, order_, xi,
                                                       nullptr, gradients_out, nullptr);
            return;
        case LagrangeTopology::Tetrahedron:
            detail::evaluate_tetrahedron_simplex_basis_to(simplex_exponents_, order_, xi,
                                                          nullptr, gradients_out, nullptr);
            return;
        case LagrangeTopology::Wedge: {
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations z_axis =
                fill_axis_scratch(scratch.axis_z, vc, dc, d2c, bw, n_axis, xi[2], AxisDeriv::ValuesAndFirst);
            const std::size_t tri_count = simplex_exponents_.size();
            scratch.tri_values.resize(tri_count);
            scratch.tri_gradient_components.resize(tri_count * 3u);
            detail::evaluate_triangle_simplex_basis_to(simplex_exponents_, order_, xi,
                                                       scratch.tri_values.data(),
                                                       scratch.tri_gradient_components.data(),
                                                       nullptr);
            for (std::size_t n = 0; n < wedge_indices_.size(); ++n) {
                const auto& index = wedge_indices_[n];
                const std::size_t tri = index[0];
                const std::size_t z = index[1];
                const Real* tri_g = scratch.tri_gradient_components.data() + tri * 3u;
                Real* g = gradients_out + n * 3u;
                g[0] = tri_g[0] * z_axis.values[z];
                g[1] = tri_g[1] * z_axis.values[z];
                g[2] = scratch.tri_values[tri] * z_axis.first[z];
            }
            return;
        }
        case LagrangeTopology::Pyramid: {
            detail::lagrange_pyramid::evaluate_gradients_to(order_, xi, gradients_out);
            return;
        }
        case LagrangeTopology::Unknown:
            break;
    }

    throw BasisEvaluationException("Unsupported element in evaluate_gradients_to",
                                   __FILE__, __LINE__, __func__);
}

void LagrangeBasis::evaluate_hessians_to(const math::Vector<Real, 3>& xi,
                                         Real* SVMP_RESTRICT hessians_out) const {
    const auto topology = static_cast<LagrangeTopology>(topology_id_);
    if (evaluate_fixed_lagrange_fast_to(topology, order_, xi, nullptr, nullptr, hessians_out)) {
        return;
    }

    const int n_axis = static_cast<int>(nodes_1d_.size());
    const Real* vc = axis_v_coeffs_.data();
    const Real* dc = axis_d_coeffs_.data();
    const Real* d2c = axis_d2_coeffs_.data();
    const Real* bw = axis_barycentric_weights_.data();
    switch (topology) {
        case LagrangeTopology::Point:
            for (std::size_t i = 0; i < 9; ++i) {
                hessians_out[i] = Real(0);
            }
            return;
        case LagrangeTopology::Line:
        case LagrangeTopology::Quadrilateral:
        case LagrangeTopology::Hexahedron: {
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations x_axis =
                fill_axis_scratch(scratch.axis_x, vc, dc, d2c, bw, n_axis, xi[0], AxisDeriv::ValuesAndFirstAndSecond);
            AxisBasisEvaluations y_axis = constant_axis_basis();
            AxisBasisEvaluations z_axis = constant_axis_basis();
            if (topology != LagrangeTopology::Line) {
                y_axis = fill_axis_scratch(scratch.axis_y, vc, dc, d2c, bw, n_axis, xi[1], AxisDeriv::ValuesAndFirstAndSecond);
            }
            if (topology == LagrangeTopology::Hexahedron) {
                z_axis = fill_axis_scratch(scratch.axis_z, vc, dc, d2c, bw, n_axis, xi[2], AxisDeriv::ValuesAndFirstAndSecond);
            }
            evaluate_tensor_product_factorized_to(tensor_indices_, x_axis, y_axis, z_axis,
                                                  nullptr, nullptr, hessians_out);
            return;
        }
        case LagrangeTopology::Triangle:
            detail::evaluate_triangle_simplex_basis_to(simplex_exponents_, order_, xi,
                                                       nullptr, nullptr, hessians_out);
            return;
        case LagrangeTopology::Tetrahedron:
            detail::evaluate_tetrahedron_simplex_basis_to(simplex_exponents_, order_, xi,
                                                          nullptr, nullptr, hessians_out);
            return;
        case LagrangeTopology::Wedge: {
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations z_axis =
                fill_axis_scratch(scratch.axis_z, vc, dc, d2c, bw, n_axis, xi[2], AxisDeriv::ValuesAndFirstAndSecond);
            const std::size_t tri_count = simplex_exponents_.size();
            scratch.tri_values.resize(tri_count);
            scratch.tri_gradient_components.resize(tri_count * 3u);
            scratch.tri_hessian_components.resize(tri_count * 9u);
            detail::evaluate_triangle_simplex_basis_to(simplex_exponents_, order_, xi,
                                                       scratch.tri_values.data(),
                                                       scratch.tri_gradient_components.data(),
                                                       scratch.tri_hessian_components.data());
            for (std::size_t n = 0; n < wedge_indices_.size(); ++n) {
                const auto& index = wedge_indices_[n];
                const std::size_t tri = index[0];
                const std::size_t z = index[1];
                const Real zv = z_axis.values[z];
                const Real zd = z_axis.first[z];
                const Real zd2 = z_axis.second[z];
                const Real* tri_g = scratch.tri_gradient_components.data() + tri * 3u;
                const Real* tri_H = scratch.tri_hessian_components.data() + tri * 9u;
                Real* H = hessians_out + n * 9u;
                H[0] = tri_H[0] * zv;
                H[4] = tri_H[4] * zv;
                H[1] = tri_H[1] * zv;
                H[3] = H[1];
                H[8] = scratch.tri_values[tri] * zd2;
                H[2] = tri_g[0] * zd;
                H[6] = H[2];
                H[5] = tri_g[1] * zd;
                H[7] = H[5];
            }
            return;
        }
        case LagrangeTopology::Pyramid: {
            detail::lagrange_pyramid::evaluate_hessians_to(order_, xi, hessians_out);
            return;
        }
        case LagrangeTopology::Unknown:
            break;
    }

    throw BasisEvaluationException("Unsupported element in evaluate_hessians_to",
                                   __FILE__, __LINE__, __func__);
}

void LagrangeBasis::evaluate_all_to(const math::Vector<Real, 3>& xi,
                                    Real* SVMP_RESTRICT values_out,
                                    Real* SVMP_RESTRICT gradients_out,
                                    Real* SVMP_RESTRICT hessians_out) const {
    const auto topology = static_cast<LagrangeTopology>(topology_id_);
    if (evaluate_fixed_lagrange_fast_to(topology, order_, xi, values_out, gradients_out, hessians_out)) {
        return;
    }

    const int n_axis = static_cast<int>(nodes_1d_.size());
    const Real* vc = axis_v_coeffs_.data();
    const Real* dc = axis_d_coeffs_.data();
    const Real* d2c = axis_d2_coeffs_.data();
    const Real* bw = axis_barycentric_weights_.data();
    switch (topology) {
        case LagrangeTopology::Point:
            values_out[0] = Real(1);
            gradients_out[0] = Real(0);
            gradients_out[1] = Real(0);
            gradients_out[2] = Real(0);
            for (std::size_t i = 0; i < 9; ++i) {
                hessians_out[i] = Real(0);
            }
            return;
        case LagrangeTopology::Line:
        case LagrangeTopology::Quadrilateral:
        case LagrangeTopology::Hexahedron: {
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations x_axis =
                fill_axis_scratch(scratch.axis_x, vc, dc, d2c, bw, n_axis, xi[0], AxisDeriv::ValuesAndFirstAndSecond);
            AxisBasisEvaluations y_axis = constant_axis_basis();
            AxisBasisEvaluations z_axis = constant_axis_basis();
            if (topology != LagrangeTopology::Line) {
                y_axis = fill_axis_scratch(scratch.axis_y, vc, dc, d2c, bw, n_axis, xi[1], AxisDeriv::ValuesAndFirstAndSecond);
            }
            if (topology == LagrangeTopology::Hexahedron) {
                z_axis = fill_axis_scratch(scratch.axis_z, vc, dc, d2c, bw, n_axis, xi[2], AxisDeriv::ValuesAndFirstAndSecond);
            }
            evaluate_tensor_product_factorized_to(tensor_indices_, x_axis, y_axis, z_axis,
                                                  values_out, gradients_out, hessians_out);
            return;
        }
        case LagrangeTopology::Triangle:
            detail::evaluate_triangle_simplex_basis_to(simplex_exponents_, order_, xi,
                                                       values_out, gradients_out, hessians_out);
            return;
        case LagrangeTopology::Tetrahedron:
            detail::evaluate_tetrahedron_simplex_basis_to(simplex_exponents_, order_, xi,
                                                          values_out, gradients_out, hessians_out);
            return;
        case LagrangeTopology::Wedge: {
            LagrangeEvaluateScratch& scratch = evaluate_scratch();
            const AxisBasisEvaluations z_axis =
                fill_axis_scratch(scratch.axis_z, vc, dc, d2c, bw, n_axis, xi[2], AxisDeriv::ValuesAndFirstAndSecond);
            const std::size_t tri_count = simplex_exponents_.size();
            scratch.tri_values.resize(tri_count);
            scratch.tri_gradient_components.resize(tri_count * 3u);
            scratch.tri_hessian_components.resize(tri_count * 9u);
            detail::evaluate_triangle_simplex_basis_to(simplex_exponents_, order_, xi,
                                                       scratch.tri_values.data(),
                                                       scratch.tri_gradient_components.data(),
                                                       scratch.tri_hessian_components.data());
            for (std::size_t n = 0; n < wedge_indices_.size(); ++n) {
                const auto& index = wedge_indices_[n];
                const std::size_t tri = index[0];
                const std::size_t z = index[1];
                const Real zv = z_axis.values[z];
                const Real zd = z_axis.first[z];
                const Real zd2 = z_axis.second[z];
                const Real tri_v = scratch.tri_values[tri];
                const Real* tri_g = scratch.tri_gradient_components.data() + tri * 3u;
                const Real* tri_H = scratch.tri_hessian_components.data() + tri * 9u;

                values_out[n] = tri_v * zv;

                Real* g = gradients_out + n * 3u;
                g[0] = tri_g[0] * zv;
                g[1] = tri_g[1] * zv;
                g[2] = tri_v * zd;

                Real* H = hessians_out + n * 9u;
                H[0] = tri_H[0] * zv;
                H[4] = tri_H[4] * zv;
                H[1] = tri_H[1] * zv;
                H[3] = H[1];
                H[8] = tri_v * zd2;
                H[2] = tri_g[0] * zd;
                H[6] = H[2];
                H[5] = tri_g[1] * zd;
                H[7] = H[5];
            }
            return;
        }
        case LagrangeTopology::Pyramid: {
            detail::lagrange_pyramid::evaluate_all_to(
                order_, xi, values_out, gradients_out, hessians_out);
            return;
        }
        case LagrangeTopology::Unknown:
            break;
    }

    throw BasisEvaluationException("Unsupported element in evaluate_all_to",
                                   __FILE__, __LINE__, __func__);
}

void LagrangeBasis::evaluate_at_quadrature_points(
    const std::vector<math::Vector<Real, 3>>& points,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    evaluate_at_quadrature_points_strided(points, points.size(), values_out, gradients_out, hessians_out);
}

void LagrangeBasis::evaluate_at_quadrature_points_strided(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    const std::size_t num_qpts = points.size();
    const std::size_t num_dofs = size();
    if (output_stride < num_qpts) {
        throw BasisConfigurationException("LagrangeBasis strided evaluation requires output_stride >= points.size()",
                                          __FILE__, __LINE__, __func__);
    }
    if (values_out == nullptr && gradients_out == nullptr && hessians_out == nullptr) {
        return;
    }

    const auto topology = static_cast<LagrangeTopology>(topology_id_);
    if (topology == LagrangeTopology::Line ||
        topology == LagrangeTopology::Quadrilateral ||
        topology == LagrangeTopology::Hexahedron) {
        evaluate_tensor_product_points_strided(topology,
                                               tensor_indices_,
                                               points,
                                               output_stride,
                                               axis_v_coeffs_.data(),
                                               axis_d_coeffs_.data(),
                                               axis_d2_coeffs_.data(),
                                               axis_barycentric_weights_.data(),
                                               static_cast<int>(nodes_1d_.size()),
                                               values_out,
                                               gradients_out,
                                               hessians_out);
        return;
    }

    if (topology == LagrangeTopology::Triangle) {
        detail::evaluate_triangle_simplex_basis_strided(
            simplex_exponents_, order_, points, output_stride, values_out, gradients_out, hessians_out);
        return;
    }

    if (topology == LagrangeTopology::Tetrahedron) {
        detail::evaluate_tetrahedron_simplex_basis_strided(
            simplex_exponents_, order_, points, output_stride, values_out, gradients_out, hessians_out);
        return;
    }

    if (topology == LagrangeTopology::Wedge) {
        evaluate_wedge_points_strided(simplex_exponents_,
                                      wedge_indices_,
                                      order_,
                                      points,
                                      output_stride,
                                      axis_v_coeffs_.data(),
                                      axis_d_coeffs_.data(),
                                      axis_d2_coeffs_.data(),
                                      axis_barycentric_weights_.data(),
                                      static_cast<int>(nodes_1d_.size()),
                                      values_out,
                                      gradients_out,
                                      hessians_out);
        return;
    }

    if (topology == LagrangeTopology::Pyramid) {
        detail::lagrange_pyramid::evaluate_at_quadrature_points_strided(
            order_, points, output_stride, values_out, gradients_out, hessians_out);
        return;
    }

    static thread_local std::vector<Real> v_tmp;
    static thread_local std::vector<Real> g_tmp;
    static thread_local std::vector<Real> h_tmp;

    if (values_out)    v_tmp.resize(num_dofs);
    if (gradients_out) g_tmp.resize(num_dofs * 3u);
    if (hessians_out)  h_tmp.resize(num_dofs * 9u);

    for (std::size_t q = 0; q < num_qpts; ++q) {
        if (values_out && gradients_out && hessians_out) {
            evaluate_all_to(points[q], v_tmp.data(), g_tmp.data(), h_tmp.data());
        } else {
            if (values_out)    evaluate_values_to(points[q], v_tmp.data());
            if (gradients_out) evaluate_gradients_to(points[q], g_tmp.data());
            if (hessians_out)  evaluate_hessians_to(points[q], h_tmp.data());
        }

        if (values_out) {
            for (std::size_t d = 0; d < num_dofs; ++d) {
                values_out[d * output_stride + q] = v_tmp[d];
            }
        }
        if (gradients_out) {
            for (std::size_t d = 0; d < num_dofs; ++d) {
                gradients_out[(d * 3u + 0u) * output_stride + q] = g_tmp[d * 3u + 0u];
                gradients_out[(d * 3u + 1u) * output_stride + q] = g_tmp[d * 3u + 1u];
                gradients_out[(d * 3u + 2u) * output_stride + q] = g_tmp[d * 3u + 2u];
            }
        }
        if (hessians_out) {
            for (std::size_t d = 0; d < num_dofs; ++d) {
                scatter_hessian_components_strided(
                    h_tmp.data() + d * 9u,
                    hessians_out + d * 9u * output_stride,
                    output_stride,
                    q);
            }
        }
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
