#include "LevelSet/LevelSetKinematicReconciliation.h"

#include "Assembly/CutIntegrationContext.h"
#include "Basis/NodeOrderingConventions.h"
#include "Dofs/EntityDofMap.h"
#include "LevelSet/LevelSetVolume.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/FunctionSpace.h"

#if FE_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace svmp::FE::level_set {
namespace {

// The endpoint geometry depends weakly on the correction (it moves the
// interface by a small fraction of the step's motion), so the fixed point
// converges in a few iterations.  These are solver controls, not model
// parameters: the result is the converged fixed point.
constexpr int kMaxGeometryIterations = 64;
constexpr Real kGeometryRelativeTolerance = 1.0e-10;

using Point = std::array<Real, 3>;

struct Collective {
#if FE_HAS_MPI
    MPI_Comm communicator{MPI_COMM_NULL};
#endif
    bool active{false};
};

[[nodiscard]] Collective collectiveFor(const dofs::DofHandler& dof_handler)
{
    Collective context;
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (initialized != 0 && finalized == 0 &&
        dof_handler.mpiComm() != MPI_COMM_NULL) {
        int size = 1;
        MPI_Comm_size(dof_handler.mpiComm(), &size);
        context.communicator = dof_handler.mpiComm();
        context.active = size > 1;
    }
#else
    (void)dof_handler;
#endif
    return context;
}

#if FE_HAS_MPI
[[nodiscard]] MPI_Datatype realType() noexcept
{
    if constexpr (std::is_same_v<Real, float>) {
        return MPI_FLOAT;
    }
    return MPI_DOUBLE;
}
#endif

void sumInPlace(const Collective& context, std::vector<Real>& values)
{
#if FE_HAS_MPI
    if (context.active && !values.empty()) {
        if (values.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "level-set kinematic reconciliation exceeds the MPI count range");
        }
        MPI_Allreduce(MPI_IN_PLACE,
                      values.data(),
                      static_cast<int>(values.size()),
                      realType(),
                      MPI_SUM,
                      context.communicator);
    }
#else
    (void)context;
    (void)values;
#endif
}

[[nodiscard]] bool allTrue(const Collective& context, bool local)
{
#if FE_HAS_MPI
    if (context.active) {
        int value = local ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_INT, MPI_MIN,
                      context.communicator);
        return value != 0;
    }
#else
    (void)context;
#endif
    return local;
}

[[nodiscard]] unsigned long long sumCount(const Collective& context,
                                          unsigned long long local)
{
#if FE_HAS_MPI
    if (context.active) {
        MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM, context.communicator);
    }
#else
    (void)context;
#endif
    return local;
}

[[nodiscard]] Point sub(const Point& a, const Point& b) noexcept
{
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

[[nodiscard]] Point cross(const Point& a, const Point& b) noexcept
{
    return {{a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]}};
}

[[nodiscard]] Real dot(const Point& a, const Point& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] Real norm(const Point& a) noexcept
{
    return std::sqrt(dot(a, a));
}

// One owned affine simplex with its level-set DOFs and nodal velocities.
// x holds the physical corner coordinates; xi the reference-element corner
// coordinates, in which the generated-interface builder cuts the cell.
struct CellTrace {
    std::array<Real, 4> g{};
    bool kinematically_exact{true};
};

struct SimplexCell {
    int corners{0};
    std::array<Point, 4> x{};
    std::array<Point, 4> xi{};
    std::array<std::size_t, 4> dof{};
    std::array<Point, 4> previous_velocity{};
    std::array<Point, 4> transported_velocity{};
    std::vector<Real> previous_coefficients;
    std::vector<Real> transported_coefficients;
    const spaces::FunctionSpace* higher_order_velocity{};
    std::shared_ptr<const quadrature::QuadratureRule> trace_rule;
    Real measure{};
    Real previous_fraction{};
    CellTrace previous_trace;
};

// Gradient of the P1 interpolant of the corner values on the simplex.
[[nodiscard]] bool simplexGradient(const SimplexCell& cell,
                                   const std::array<Real, 4>& values,
                                   Point& gradient) noexcept
{
    if (cell.corners == 3) {
        const auto e1 = sub(cell.x[1], cell.x[0]);
        const auto e2 = sub(cell.x[2], cell.x[0]);
        const Real det = e1[0] * e2[1] - e1[1] * e2[0];
        if (!(std::abs(det) > Real{0.0}) || !std::isfinite(det)) {
            return false;
        }
        const Real d1 = values[1] - values[0];
        const Real d2 = values[2] - values[0];
        gradient = {{(d1 * e2[1] - d2 * e1[1]) / det,
                     (e1[0] * d2 - e2[0] * d1) / det,
                     Real{0.0}}};
        return true;
    }
    const auto e1 = sub(cell.x[1], cell.x[0]);
    const auto e2 = sub(cell.x[2], cell.x[0]);
    const auto e3 = sub(cell.x[3], cell.x[0]);
    const auto c23 = cross(e2, e3);
    const auto c31 = cross(e3, e1);
    const auto c12 = cross(e1, e2);
    const Real det = dot(e1, c23);
    if (!(std::abs(det) > Real{0.0}) || !std::isfinite(det)) {
        return false;
    }
    const Real d1 = values[1] - values[0];
    const Real d2 = values[2] - values[0];
    const Real d3 = values[3] - values[0];
    for (std::size_t k = 0; k < 3u; ++k) {
        gradient[k] = (d1 * c23[k] + d2 * c31[k] + d3 * c12[k]) / det;
    }
    return true;
}

// Nodal interface integrals of one cell for one geometry:
//   g_i += int N_i / |grad phi|,  f_i += int N_i (w . grad phi) / |grad phi|,
//   m_i += int N_i d / |grad phi|   (d = transported rate).
struct InterfaceSums {
    std::vector<Real> g;
    std::vector<Real> f;
    std::vector<Real> m;

    explicit InterfaceSums(std::size_t n) : g(n, 0.0), f(n, 0.0), m(n, 0.0) {}

    void reduce(const Collective& context)
    {
        std::vector<Real> packed;
        packed.reserve(3u * g.size());
        packed.insert(packed.end(), g.begin(), g.end());
        packed.insert(packed.end(), f.begin(), f.end());
        packed.insert(packed.end(), m.begin(), m.end());
        sumInPlace(context, packed);
        const auto n = g.size();
        std::copy(packed.begin(), packed.begin() + static_cast<std::ptrdiff_t>(n), g.begin());
        std::copy(packed.begin() + static_cast<std::ptrdiff_t>(n),
                  packed.begin() + static_cast<std::ptrdiff_t>(2u * n), f.begin());
        std::copy(packed.begin() + static_cast<std::ptrdiff_t>(2u * n), packed.end(), m.begin());
    }
};

// Accumulate the interface integrals of one cell.  Returns true when the
// cell is cut by {phi = c}.
bool accumulateCell(const SimplexCell& cell,
                    std::span<const Real> phi,
                    std::span<const Real> rate,
                    bool previous,
                    Real isovalue,
                    Real rate_roundoff,
                    InterfaceSums& sums,
                    CellTrace& trace)
{
    const int n = cell.corners;
    std::array<Real, 4> s{};
    std::array<Real, 4> values{};
    int negative = 0;
    for (int i = 0; i < n; ++i) {
        values[static_cast<std::size_t>(i)] = phi[cell.dof[static_cast<std::size_t>(i)]];
        s[static_cast<std::size_t>(i)] = values[static_cast<std::size_t>(i)] - isovalue;
        negative += s[static_cast<std::size_t>(i)] < Real{0.0} ? 1 : 0;
    }
    if (negative == 0 || negative == n) {
        return false;
    }
    Point gradient{};
    if (!simplexGradient(cell, values, gradient)) {
        throw std::runtime_error(
            "level-set kinematic reconciliation found a degenerate cut cell");
    }
    const Real gradient_norm = norm(gradient);
    if (!(gradient_norm > Real{0.0}) || !std::isfinite(gradient_norm)) {
        throw std::runtime_error(
            "level-set kinematic reconciliation found a cut cell without a finite level-set gradient");
    }

    // Cut points on the sign-changing edges, as barycentric coordinates.
    using Bary = std::array<Real, 4>;
    const auto edgePoint = [&](int a, int b) {
        const Real sa = s[static_cast<std::size_t>(a)];
        const Real sb = s[static_cast<std::size_t>(b)];
        Real t = sa / (sa - sb);
        t = std::clamp(t, Real{0.0}, Real{1.0});
        Bary lambda{};
        lambda[static_cast<std::size_t>(a)] = Real{1.0} - t;
        lambda[static_cast<std::size_t>(b)] = t;
        return lambda;
    };
    const auto position = [&](const Bary& lambda) {
        Point p{};
        for (int i = 0; i < n; ++i) {
            for (std::size_t k = 0; k < 3u; ++k) {
                p[k] += lambda[static_cast<std::size_t>(i)] * cell.x[static_cast<std::size_t>(i)][k];
            }
        }
        return p;
    };
    const auto addPoint = [&](const Bary& lambda, Real weight) {
        const Real scaled = weight / gradient_norm;
        Real advective = 0.0;
        Real rate_value = 0.0;
        const auto& velocity = previous ? cell.previous_velocity : cell.transported_velocity;
        for (int j = 0; j < n; ++j) {
            const auto lj = lambda[static_cast<std::size_t>(j)];
            advective += lj * dot(velocity[static_cast<std::size_t>(j)], gradient);
            rate_value += lj * rate[cell.dof[static_cast<std::size_t>(j)]];
        }
        if (cell.higher_order_velocity != nullptr) {
            const spaces::FunctionSpace::Value xi{lambda[1], lambda[2], lambda[3]};
            const auto& coefficients = previous ? cell.previous_coefficients : cell.transported_coefficients;
            const auto value = cell.higher_order_velocity->evaluate(xi, coefficients);
            advective = value[0] * gradient[0] + value[1] * gradient[1] + value[2] * gradient[2];
        }
        if (!std::isfinite(advective) || !std::isfinite(rate_value)) {
            throw std::runtime_error("non-finite interface velocity or level-set rate");
        }
        const Real evaluation_roundoff = rate_roundoff +
            Real{64} * std::numeric_limits<Real>::epsilon() * (std::abs(advective) + std::abs(rate_value));
        trace.kinematically_exact = trace.kinematically_exact &&
                                    std::abs(advective + rate_value) <= evaluation_roundoff;
        for (int i = 0; i < n; ++i) {
            const auto li = lambda[static_cast<std::size_t>(i)];
            const auto dof = cell.dof[static_cast<std::size_t>(i)];
            trace.g[static_cast<std::size_t>(i)] += scaled * li;
            sums.g[dof] += scaled * li;
            sums.f[dof] += scaled * li * advective;
            sums.m[dof] += scaled * li * rate_value;
        }
    };
    const auto midpoint = [&](const Bary& a, const Bary& b) {
        Bary m{};
        for (std::size_t i = 0; i < 4u; ++i) {
            m[i] = Real{0.5} * (a[i] + b[i]);
        }
        return m;
    };
    // Exact for quadratics on a triangle: the three edge midpoints.
    const auto addTriangle = [&](const Bary& a, const Bary& b, const Bary& c) {
        const Real area = Real{0.5} * norm(cross(sub(position(b), position(a)),
                                                 sub(position(c), position(a))));
        if (!(area > Real{0.0})) {
            return;
        }
        if (cell.trace_rule) {
            for (std::size_t q = 0; q < cell.trace_rule->num_points(); ++q) {
                const auto xi = cell.trace_rule->point(q);
                Bary p{};
                for (std::size_t j = 0; j < p.size(); ++j) {
                    p[j] = a[j] + xi[0] * (b[j] - a[j]) + xi[1] * (c[j] - a[j]);
                }
                addPoint(p, Real{2} * area * cell.trace_rule->weight(q));
            }
            return;
        }
        addPoint(midpoint(a, b), area / Real{3.0});
        addPoint(midpoint(b, c), area / Real{3.0});
        addPoint(midpoint(c, a), area / Real{3.0});
    };

    std::array<int, 4> neg{};
    std::array<int, 4> pos{};
    int n_neg = 0;
    int n_pos = 0;
    for (int i = 0; i < n; ++i) {
        if (s[static_cast<std::size_t>(i)] < Real{0.0}) {
            neg[static_cast<std::size_t>(n_neg++)] = i;
        } else {
            pos[static_cast<std::size_t>(n_pos++)] = i;
        }
    }

    if (n == 3) {
        // One isolated corner; the segment joins its two edges.
        const int lone = n_neg == 1 ? neg[0] : pos[0];
        std::array<int, 2> others{};
        int k = 0;
        for (int i = 0; i < 3; ++i) {
            if (i != lone) {
                others[static_cast<std::size_t>(k++)] = i;
            }
        }
        const auto a = edgePoint(lone, others[0]);
        const auto b = edgePoint(lone, others[1]);
        const Real length = norm(sub(position(b), position(a)));
        if (length > Real{0.0}) {
            if (cell.trace_rule) {
                for (std::size_t q = 0; q < cell.trace_rule->num_points(); ++q) {
                    const Real t = Real{0.5} * (Real{1} + cell.trace_rule->point(q)[0]);
                    Bary p{};
                    for (std::size_t j = 0; j < p.size(); ++j) {
                        p[j] = a[j] + t * (b[j] - a[j]);
                    }
                    addPoint(p, Real{0.5} * length * cell.trace_rule->weight(q));
                }
                return true;
            }
            // Simpson's rule, exact for quadratics along the segment.
            addPoint(a, length / Real{6.0});
            addPoint(midpoint(a, b), Real{4.0} * length / Real{6.0});
            addPoint(b, length / Real{6.0});
        }
        return true;
    }

    if (n_neg == 1 || n_pos == 1) {
        const int lone = n_neg == 1 ? neg[0] : pos[0];
        std::array<Bary, 3> p{};
        int k = 0;
        for (int i = 0; i < 4; ++i) {
            if (i != lone) {
                p[static_cast<std::size_t>(k++)] = edgePoint(lone, i);
            }
        }
        addTriangle(p[0], p[1], p[2]);
        return true;
    }
    // Two corners on each side: a planar quadrilateral with the cyclic
    // vertex order (a,p), (a,q), (b,q), (b,p).
    const auto ap = edgePoint(neg[0], pos[0]);
    const auto aq = edgePoint(neg[0], pos[1]);
    const auto bq = edgePoint(neg[1], pos[1]);
    const auto bp = edgePoint(neg[1], pos[0]);
    addTriangle(ap, aq, bq);
    addTriangle(ap, bq, bp);
    return true;
}

// Classification of the P1 cut of one simplex as the generated-interface
// builder and the free-surface geometry snapshot see it: the interface
// fragment is collapsed (measure <= tolerance), nearly tangent
// (<= sqrt(tolerance)) or regular, and each side's volume rule is pruned when
// its volume fraction is positive but below the minimum retained fraction.
// The measure is the segment length in 2D and the polygon area in 3D, taken
// in the reference-element coordinates of the cell: the builder cuts every
// cell on its reference element (LevelSetInterfaceLifecycle), so its measure
// thresholds are reference measures.  A physical measure differs from it by
// the cell Jacobian (about h^2 for a 3D polygon), so it would guard thresholds
// the builder does not have and miss the ones it has.  The volume fractions
// are ratios and do not depend on the frame.
[[nodiscard]] int cutClass(const SimplexCell& cell,
                           std::span<const Real> phi,
                           Real isovalue,
                           Real tolerance,
                           Real minimum_fraction,
                           Real* cut_negative_fraction = nullptr) noexcept
{
    const int n = cell.corners;
    std::array<Real, 4> s{};
    std::array<int, 4> neg{};
    std::array<int, 4> pos{};
    int n_neg = 0;
    int n_pos = 0;
    for (int i = 0; i < n; ++i) {
        s[static_cast<std::size_t>(i)] = phi[cell.dof[static_cast<std::size_t>(i)]] - isovalue;
        if (s[static_cast<std::size_t>(i)] < Real{0.0}) {
            neg[static_cast<std::size_t>(n_neg++)] = i;
        } else {
            pos[static_cast<std::size_t>(n_pos++)] = i;
        }
    }
    if (n_neg == 0 || n_pos == 0) {
        if (cut_negative_fraction) {
            *cut_negative_fraction = n_neg == 0 ? Real{0} : Real{1};
        }
        return 3;
    }
    const auto fraction = [&](int a, int b) {
        const Real sa = s[static_cast<std::size_t>(a)];
        return std::clamp(sa / (sa - s[static_cast<std::size_t>(b)]), Real{0.0}, Real{1.0});
    };
    const auto point = [&](int a, int b) {
        const Real t = fraction(a, b);
        Point p{};
        for (std::size_t k = 0; k < 3u; ++k) {
            p[k] = (Real{1.0} - t) * cell.xi[static_cast<std::size_t>(a)][k] +
                   t * cell.xi[static_cast<std::size_t>(b)][k];
        }
        return p;
    };
    const auto triangleArea = [](const Point& a, const Point& b, const Point& c) {
        return Real{0.5} * norm(cross(sub(b, a), sub(c, a)));
    };
    const auto tetVolume = [](const Point& a, const Point& b, const Point& c, const Point& d) {
        return std::abs(dot(sub(b, a), cross(sub(c, a), sub(d, a)))) / Real{6.0};
    };
    Real measure = 0.0;
    Real negative_fraction = 0.0;
    if (n == 3) {
        const int lone = n_neg == 1 ? neg[0] : pos[0];
        std::array<int, 2> others{};
        int k = 0;
        for (int i = 0; i < 3; ++i) {
            if (i != lone) {
                others[static_cast<std::size_t>(k++)] = i;
            }
        }
        measure = norm(sub(point(lone, others[1]), point(lone, others[0])));
        const Real corner = fraction(lone, others[0]) * fraction(lone, others[1]);
        negative_fraction = n_neg == 1 ? corner : Real{1.0} - corner;
    } else if (n_neg == 1 || n_pos == 1) {
        const int lone = n_neg == 1 ? neg[0] : pos[0];
        std::array<Point, 3> p{};
        Real corner = 1.0;
        int k = 0;
        for (int i = 0; i < 4; ++i) {
            if (i != lone) {
                p[static_cast<std::size_t>(k++)] = point(lone, i);
                corner *= fraction(lone, i);
            }
        }
        measure = triangleArea(p[0], p[1], p[2]);
        negative_fraction = n_neg == 1 ? corner : Real{1.0} - corner;
    } else {
        const auto ap = point(neg[0], pos[0]);
        const auto aq = point(neg[0], pos[1]);
        const auto bq = point(neg[1], pos[1]);
        const auto bp = point(neg[1], pos[0]);
        measure = triangleArea(ap, aq, bq) + triangleArea(ap, bq, bp);
        // The negative region is the prism (a, ap, aq) -> (b, bp, bq).
        const auto& a = cell.xi[static_cast<std::size_t>(neg[0])];
        const auto& b = cell.xi[static_cast<std::size_t>(neg[1])];
        const Real volume = tetVolume(cell.xi[0], cell.xi[1], cell.xi[2], cell.xi[3]);
        const Real negative = tetVolume(a, ap, aq, b) + tetVolume(ap, aq, b, bp) +
                              tetVolume(aq, b, bp, bq);
        negative_fraction = volume > Real{0.0} ? negative / volume : Real{0.0};
    }
    if (cut_negative_fraction) {
        *cut_negative_fraction = negative_fraction;
    }
    int code = measure <= tolerance ? 0 : (measure <= std::sqrt(tolerance) ? 1 : 2);
    const Real positive_fraction = Real{1.0} - negative_fraction;
    if (negative_fraction > Real{0.0} && negative_fraction < minimum_fraction) {
        code += 4;
    }
    if (positive_fraction > Real{0.0} && positive_fraction < minimum_fraction) {
        code += 8;
    }
    return code;
}

void maxInPlace(const Collective& context, std::vector<Real>& values)
{
#if FE_HAS_MPI
    if (context.active && !values.empty()) {
        MPI_Allreduce(MPI_IN_PLACE,
                      values.data(),
                      static_cast<int>(values.size()),
                      realType(),
                      MPI_MAX,
                      context.communicator);
    }
#else
    (void)context;
    (void)values;
#endif
}

[[nodiscard]] int signClass(Real value, Real isovalue, Real tolerance) noexcept
{
    const Real s = value - isovalue;
    if (s < -tolerance) {
        return -1;
    }
    if (s > tolerance) {
        return 1;
    }
    return 0;
}

// A read-only endpoint measurement stops after the first trace integration.
LevelSetKinematicReconciliationResult integrateKinematicStep(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const dofs::DofHandler& velocity_dofs,
    Real isovalue,
    Real tolerance,
    Real dt,
    std::span<const Real> previous_level_set,
    std::span<const Real> transported_level_set,
    std::span<const Real> previous_velocity,
    std::span<const Real> transported_velocity,
    std::vector<Real>* reconciled_level_set,
    const spaces::FunctionSpace* velocity_space)
{
    LevelSetKinematicReconciliationResult result;
    const auto context = collectiveFor(level_set_dofs);
    const auto n = static_cast<std::size_t>(level_set_dofs.getNumDofs());
    const auto n_velocity = static_cast<std::size_t>(velocity_dofs.getNumDofs());
    const int dimension = mesh.dimension();

    const Real minimum_fraction = assembly::CutIntegrationContext::minGeneratedCutVolumeFraction();
    // ---- validation (collective) -------------------------------------------
    std::string local_failure;
    if (!(dt > Real{0.0}) || !std::isfinite(dt) || !std::isfinite(isovalue) ||
        !(tolerance >= Real{0.0}) || !std::isfinite(tolerance)) {
        local_failure = "requires a positive finite time step, a finite isovalue and a nonnegative finite tolerance";
    } else if (dimension != 2 && dimension != 3) {
        local_failure = "requires a 2D or 3D mesh";
    } else if (previous_level_set.size() != n || transported_level_set.size() != n ||
               previous_velocity.size() != n_velocity ||
               transported_velocity.size() != n_velocity) {
        local_failure = "received coefficient spans that do not match the field layouts";
    } else if (velocity_space &&
               (velocity_space->value_dimension() < dimension ||
                (velocity_space->continuity() != Continuity::C0 && velocity_space->continuity() != Continuity::L2))) {
        local_failure = "requires a component-valued physical velocity";
    } else if (level_set_dofs.getEntityDofMap() == nullptr ||
               velocity_dofs.getEntityDofMap() == nullptr) {
        local_failure = "requires vertex-nodal level-set and velocity layouts";
    }
    for (std::size_t i = 0; local_failure.empty() && i < n; ++i) {
        if (!std::isfinite(previous_level_set[i]) || !std::isfinite(transported_level_set[i])) {
            local_failure = "received non-finite level-set coefficients";
        }
    }
    for (std::size_t i = 0; local_failure.empty() && i < n_velocity; ++i) {
        if (!std::isfinite(previous_velocity[i]) || !std::isfinite(transported_velocity[i])) {
            local_failure = "received non-finite velocity coefficients";
        }
    }

    // ---- gather owned simplices ---------------------------------------------
    std::vector<SimplexCell> cells;
    if (local_failure.empty()) {
        try {
            const auto* phi_map = level_set_dofs.getEntityDofMap();
            const auto* velocity_map = velocity_dofs.getEntityDofMap();
            const ElementType expected =
                dimension == 2 ? ElementType::Triangle3 : ElementType::Tetra4;
            const int corners = dimension + 1;
            std::vector<GlobalIndex> nodes;
            std::vector<std::array<Real, 3>> coordinates;
            mesh.forEachOwnedCell([&](GlobalIndex cell_id) {
                if (mesh.getCellType(cell_id) != expected ||
                    mesh.getCellGeometryOrder(cell_id) > 1) {
                    throw std::invalid_argument(
                        "supports only affine Triangle3 (2D) and Tetra4 (3D) cells");
                }
                mesh.getCellNodes(cell_id, nodes);
                mesh.getCellCoordinates(cell_id, coordinates);
                if (nodes.size() < static_cast<std::size_t>(corners) ||
                    coordinates.size() < static_cast<std::size_t>(corners)) {
                    throw std::invalid_argument("found incomplete cell geometry");
                }
                SimplexCell cell;
                cell.corners = corners;
                const auto phi_cell = level_set_dofs.getDofMap().getCellDofs(cell_id);
                if (phi_cell.size() != static_cast<std::size_t>(corners)) {
                    throw std::invalid_argument("requires a P1 scalar level-set space");
                }
                const auto velocity_cell = velocity_dofs.getDofMap().getCellDofs(cell_id);
                const auto vertex_components = velocity_map->getVertexDofs(nodes.front()).size();
                if (vertex_components < static_cast<std::size_t>(dimension) ||
                    velocity_cell.size() != vertex_components * static_cast<std::size_t>(corners) ||
                    (velocity_space && (velocity_space->polynomial_order(cell_id) != 1 ||
                                        velocity_space->continuity() != Continuity::C0))) {
                    if (!velocity_space || velocity_space->is_variable_order() ||
                        velocity_cell.size() != velocity_space->dofs_per_element(cell_id)) {
                        throw std::invalid_argument("higher-order velocity requires its matching FE space");
                    }
                    cell.higher_order_velocity = velocity_space;
                    // Besides N_i times velocity, the consistency test needs
                    // enough points to distinguish a polynomial from zero.
                    // Exact integration of its square gives that property.
                    const int degree = 2 * std::max(1, velocity_space->polynomial_order(cell_id));
                    cell.trace_rule = quadrature::QuadratureFactory::create_pinned(
                        dimension == 2 ? ElementType::Line2 : ElementType::Triangle3, degree);
                    if (cell.trace_rule->order() < degree) {
                        throw std::invalid_argument("insufficient quadrature for the velocity trace");
                    }
                    for (const auto dof : velocity_cell) {
                        if (dof < 0 || static_cast<std::size_t>(dof) >= n_velocity) {
                            throw std::invalid_argument("invalid cell velocity DOF");
                        }
                        cell.previous_coefficients.push_back(previous_velocity[dof]);
                        cell.transported_coefficients.push_back(transported_velocity[dof]);
                    }
                }
                for (int i = 0; i < corners; ++i) {
                    const auto ii = static_cast<std::size_t>(i);
                    cell.x[ii] = coordinates[ii];
                    const auto reference =
                        basis::ReferenceNodeLayout::get_node_coords(expected, ii);
                    cell.xi[ii] = {{reference[0], reference[1], reference[2]}};
                    const auto phi_dofs = phi_map->getVertexDofs(nodes[ii]);
                    if (phi_dofs.size() != 1u || phi_dofs.front() < 0 ||
                        static_cast<std::size_t>(phi_dofs.front()) >= n) {
                        throw std::invalid_argument(
                            "requires exactly one scalar level-set DOF per vertex");
                    }
                    cell.dof[ii] = static_cast<std::size_t>(phi_dofs.front());
                    if (cell.higher_order_velocity) {
                        continue;
                    }
                    const auto velocity_vertex_dofs = velocity_map->getVertexDofs(nodes[ii]);
                    if (velocity_vertex_dofs.size() < static_cast<std::size_t>(dimension)) {
                        throw std::invalid_argument(
                            "requires one velocity DOF per spatial component at every vertex");
                    }
                    for (int d = 0; d < dimension; ++d) {
                        const auto dof = velocity_vertex_dofs[static_cast<std::size_t>(d)];
                        if (dof < 0 || static_cast<std::size_t>(dof) >= n_velocity) {
                            throw std::invalid_argument("found a velocity DOF outside its field");
                        }
                        cell.previous_velocity[ii][static_cast<std::size_t>(d)] =
                            previous_velocity[static_cast<std::size_t>(dof)];
                        cell.transported_velocity[ii][static_cast<std::size_t>(d)] =
                            transported_velocity[static_cast<std::size_t>(dof)];
                    }
                }
                const auto ab = sub(cell.x[1], cell.x[0]);
                const auto ac = sub(cell.x[2], cell.x[0]);
                cell.measure = dimension == 2 ? norm(cross(ab, ac)) / Real{2}
                    : std::abs(dot(cross(ab, ac), sub(cell.x[3], cell.x[0]))) / Real{6};
                (void)cutClass(cell, previous_level_set, isovalue, tolerance,
                               minimum_fraction, &cell.previous_fraction);
                cells.push_back(std::move(cell));
            });
        } catch (const std::exception& error) {
            local_failure = error.what();
        }
    }
    if (!allTrue(context, local_failure.empty())) {
        result.diagnostic = local_failure.empty()
                                ? "another rank failed level-set kinematic reconciliation validation"
                                : "level-set kinematic reconciliation " + local_failure;
        return result;
    }

    // ---- previous geometry and transported rate -----------------------------
    std::vector<Real> rate(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        rate[i] = (transported_level_set[i] - previous_level_set[i]) / dt;
    }
    Real coefficient_scale = 0;
    for (const auto value : transported_level_set) {
        coefficient_scale = std::max(coefficient_scale, std::abs(value - isovalue));
    }
    // Field slices are replicated. Use the communicator maximum explicitly so
    // every rank takes the same stopping and damping decisions.
    std::vector<Real> scales{coefficient_scale};
    maxInPlace(context, scales);
    const Real rounding_floor = Real{64} * std::numeric_limits<Real>::epsilon() * scales.front();
    const Real rate_roundoff = rounding_floor / dt;
    InterfaceSums previous(n);
    unsigned long long previous_cut_cells = 0u;
    try {
        for (auto& cell : cells) {
            if (accumulateCell(cell, previous_level_set, rate,
                               true, isovalue, rate_roundoff, previous, cell.previous_trace)) {
                ++previous_cut_cells;
            }
        }
    } catch (const std::exception& error) {
        local_failure = error.what();
    }
    if (!allTrue(context, local_failure.empty())) {
        result.diagnostic = local_failure.empty()
                                ? "another rank failed the previous-geometry interface integrals"
                                : "level-set kinematic reconciliation " + local_failure;
        return result;
    }
    previous.reduce(context);
    result.previous_interface_cells =
        static_cast<std::size_t>(sumCount(context, previous_cut_cells));

    LevelSetVolumeOptions volume_options{};
    volume_options.isovalue = isovalue;
    if (tolerance > Real{0.0}) {
        volume_options.tolerance = tolerance;
    }
    if (!reconciled_level_set) {
        const auto volume = computeLevelSetCutCellVolume(mesh, level_set_dofs, volume_options, previous_level_set);
        result.success = volume.success;
        result.diagnostic = volume.diagnostic;
        result.previous_negative_volume = volume.negative_volume;
        for (const auto flux : previous.f) {
            result.previous_interface_flux += flux;
        }
        return result;
    }

    // ---- fixed point on the reconciled endpoint geometry --------------------
    std::vector<Real> iterate(transported_level_set.begin(), transported_level_set.end());
    std::vector<Real> candidate(n, 0.0);
    std::vector<int> transported_class(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        transported_class[i] = signClass(transported_level_set[i], isovalue, tolerance);
    }
    Real current_flux = 0.0;
    unsigned long long current_cut_cells = 0u;
    // Cut degeneracy class of every owned cell in the transported state; the
    // reconciliation must leave every class unchanged.
    std::vector<int> transported_cut_class(cells.size(), 3);
    for (std::size_t c = 0; c < cells.size(); ++c) {
        transported_cut_class[c] = cutClass(
            cells[c], transported_level_set, isovalue, tolerance, minimum_fraction);
    }
    std::size_t skipped = 0u;
    std::size_t redistributed = 0u;
    std::size_t frozen = 0u;
    // For each simplex, account locally for the difference between its exact
    // swept volume and the trapezoidal shape derivative. Distribute the scalar
    // defect to the same nodal interface tests by their integrated weights.
    // Exactly advected traces keep their original coefficients and their
    // legitimate temporal quadrature remainder.
    std::vector<bool> supported_cell(cells.size());
    Real temporal_volume_remainder = 0;
    auto evaluate_endpoint = [&](std::span<const Real> endpoint, InterfaceSums& current) {
        unsigned long long local_cut_cells = 0;
        std::vector<Real> remainder(1, 0);
        try {
            for (std::size_t c = 0; c < cells.size(); ++c) {
                const auto& cell = cells[c];
                CellTrace trace;
                if (accumulateCell(cell, endpoint, rate, false, isovalue, rate_roundoff, current, trace)) {
                    ++local_cut_cells;
                }
                Real fraction = 0;
                (void)cutClass(cell, endpoint, isovalue, tolerance, minimum_fraction, &fraction);
                Real defect = cell.measure * (fraction - cell.previous_fraction);
                Real weight = 0;
                for (int j = 0; j < cell.corners; ++j) {
                    const Real g = Real{0.5} * (cell.previous_trace.g[j] + trace.g[j]);
                    defect += g * (endpoint[cell.dof[j]] - previous_level_set[cell.dof[j]]);
                    weight += g;
                }
                supported_cell[c] = weight > 0;
                if (trace.kinematically_exact && cell.previous_trace.kinematically_exact && weight > 0) {
                    remainder[0] += defect;
                } else if (weight > 0) {
                    for (int j = 0; j < cell.corners; ++j) {
                        current.m[cell.dof[j]] -= defect * (cell.previous_trace.g[j] + trace.g[j]) / (dt * weight);
                    }
                }
            }
        } catch (const std::exception& error) {
            local_failure = error.what();
        }
        if (!allTrue(context, local_failure.empty())) {
            result.diagnostic = local_failure.empty() ? "another rank failed the endpoint interface integrals"
                                                      : "level-set kinematic reconciliation " + local_failure;
            return false;
        }
        sumInPlace(context, remainder);
        temporal_volume_remainder = remainder.front();
        current.reduce(context);
        current_cut_cells = sumCount(context, local_cut_cells);
        current_flux = 0.0;
        for (const auto value : current.f) {
            current_flux += value;
        }

        return true;
    };
    Real prior_change = std::numeric_limits<Real>::infinity();
    Real correction_volume = 0;
    Real unplaced_volume = 0;
    for (int iteration = 0; iteration < kMaxGeometryIterations; ++iteration) {
        InterfaceSums current(n);
        if (!evaluate_endpoint(iterate, current)) {
            return result;
        }
        if (iteration == 0) {
            result.transported_interface_flux = current_flux;
        }
        // Lumped correction of every node that carries interface measure.
        // A node may not change its sign class and may not move toward the
        // isovalue by more than half of its transported distance: the cut
        // points of its edges then keep their order and stay away from the
        // vertex, so neither the cut topology nor its degeneracy class
        // changes.  The limited part of its correction is moved to the
        // available tests on its connected cut-cell patch.
        const auto admissible = [&](std::size_t i, Real proposed) {
            const Real distance = transported_level_set[i] - isovalue;
            const Real moved = distance + proposed;
            if (std::isfinite(moved) &&
                signClass(transported_level_set[i] + proposed, isovalue, tolerance) == transported_class[i] &&
                std::abs(moved) >= Real{0.5} * std::abs(distance)) {
                return proposed;
            }
            const Real halfway = Real{-0.5} * distance;
            return signClass(transported_level_set[i] + halfway, isovalue, tolerance) ==
                           transported_class[i]
                       ? halfway
                       : Real{0.0};
        };
        unplaced_volume = 0;
        skipped = 0u;
        redistributed = 0u;
        std::vector<Real> delta(n, 0.0);
        std::vector<Real> gbar(n, 0.0);
        std::vector<Real> deficit(n, 0.0);
        std::vector<char> eligible(n, 0);
        bool any_limited = false;
        for (std::size_t i = 0; i < n; ++i) {
            gbar[i] = Real{0.5} * (previous.g[i] + current.g[i]);
            if (!(gbar[i] > Real{0.0})) {
                continue;
            }
            const Real residual = Real{0.5} * (previous.m[i] + current.m[i]) +
                                  Real{0.5} * (previous.f[i] + current.f[i]);
            const Real step = -dt * residual / gbar[i];
            if (!std::isfinite(step)) {
                unplaced_volume = std::numeric_limits<Real>::infinity();
                ++skipped;
                continue;
            }
            if (transported_class[i] == 0) {
                // A pinned interface vertex carries flux even though its
                // coefficient cannot move. Its other trace nodes inherit it.
                deficit[i] = gbar[i] * step;
                any_limited = any_limited || deficit[i] != 0;
                continue;
            }
            delta[i] = admissible(i, step);
            if (delta[i] == step) {
                eligible[i] = 1;
            } else {
                deficit[i] = gbar[i] * (step - delta[i]);
                any_limited = true;
            }
        }
        // Scatter a blocked test's signed volume to the other active tests
        // on its cut cells. Count shared cells as in the lumped one-ring
        // projection. A share with no local recipient is retained for the
        // wider connected patch below.
        std::vector<Real> frozen_flag(n, 0);
        std::vector<Real> unplaced(n, 0);
        const auto scatter = [&](const std::vector<Real>& shares) {
            std::vector<Real> patch(n, 0), added(n, 0);
            const auto visit_receivers = [&](const auto& add) {
                for (std::size_t c = 0; c < cells.size(); ++c) {
                    if (!supported_cell[c]) continue;
                    const auto& cell = cells[c];
                    for (int a = 0; a < cell.corners; ++a) {
                        const auto i = cell.dof[a];
                        if (shares[i] == 0) continue;
                        for (int b = 0; b < cell.corners; ++b) {
                            const auto j = cell.dof[b];
                            if (i != j && eligible[j] && !frozen_flag[j]) add(i, j);
                        }
                    }
                }
            };
            visit_receivers([&](auto i, auto j) { patch[i] += gbar[j]; });
            sumInPlace(context, patch);
            visit_receivers([&](auto i, auto j) { added[j] += shares[i] / patch[i]; });
            sumInPlace(context, added);
            for (std::size_t i = 0; i < n; ++i) {
                if (shares[i] == 0) continue;
                if (patch[i] > 0) ++redistributed;
                else unplaced[i] += shares[i];
            }
            return added;
        };
        const auto limit_addition = [&](std::vector<Real>& added) {
            for (std::size_t i = 0; i < n; ++i) {
                if (added[i] == 0) continue;
                const Real proposed = delta[i] + added[i];
                const Real accepted = admissible(i, proposed);
                if (proposed != accepted) {
                    unplaced[i] += gbar[i] * (proposed - accepted);
                    added[i] = accepted - delta[i];
                }
            }
        };
        if (any_limited) {
            auto added = scatter(deficit);
            limit_addition(added);
            for (std::size_t i = 0; i < n; ++i) delta[i] += added[i];
        }

        // Remove only the proposed changes on cells whose builder class would
        // change. Repeat because adjacent cells share coefficients.
        std::vector<Real> trial(n);
        const auto protect_cut_classes = [&](const std::vector<Real>& base, std::vector<Real>& added,
                                             const auto& remove) {
            for (;;) {
                for (std::size_t i = 0; i < n; ++i) {
                    trial[i] = transported_level_set[i] + base[i] + added[i];
                }
                std::vector<Real> strip(n, 0);
                for (std::size_t c = 0; c < cells.size(); ++c) {
                    const auto& cell = cells[c];
                    bool touched = false;
                    for (int a = 0; a < cell.corners; ++a) touched = touched || added[cell.dof[a]] != 0;
                    if (!touched || cutClass(cell, trial, isovalue, tolerance, minimum_fraction) ==
                                        transported_cut_class[c]) continue;
                    for (int a = 0; a < cell.corners; ++a) strip[cell.dof[a]] = 1;
                }
                maxInPlace(context, strip);
                bool changed = false;
                for (std::size_t i = 0; i < n; ++i) {
                    if (!strip[i] || added[i] == 0) continue;
                    remove(i, gbar[i] * added[i]);
                    added[i] = 0;
                    changed = true;
                }
                if (!changed) return;
            }
        };
        frozen = 0;
        std::vector<Real> dropped(n, 0);
        const std::vector<Real> zero(n, 0);
        bool any_frozen = false;
        protect_cut_classes(zero, delta, [&](auto i, Real share) {
            dropped[i] += share;
            frozen_flag[i] = 1;
            any_frozen = true;
        });
        if (any_frozen) {
            auto added = scatter(dropped);
            limit_addition(added);
            protect_cut_classes(delta, added, [&](auto i, Real share) { unplaced[i] += share; });
            for (std::size_t i = 0; i < n; ++i) delta[i] += added[i];
        }

        // A frozen sliver can surround one of its vertices with other frozen
        // vertices. Carry its unplaced volume outward, one cut-cell ring at a
        // time, until it reaches an admissible trace. Never cross an uncut cell
        // or change the existing correction of a frozen node.
        for (std::size_t source = 0; source < n; ++source) {
            Real remaining = unplaced[source];
            if (remaining == 0) continue;
            std::vector<Real> seen(n, 0), front(n, 0);
            seen[source] = front[source] = 1;
            while (remaining != 0) {
                std::vector<Real> next(n, 0);
                for (std::size_t c = 0; c < cells.size(); ++c) {
                    if (!supported_cell[c]) continue;
                    const auto& cell = cells[c];
                    bool touches = false;
                    for (int a = 0; a < cell.corners; ++a) touches = touches || front[cell.dof[a]] != 0;
                    if (!touches) continue;
                    for (int a = 0; a < cell.corners; ++a) {
                        const auto i = cell.dof[a];
                        if (!seen[i] && gbar[i] > 0) next[i] = 1;
                    }
                }
                maxInPlace(context, next);
                if (std::none_of(next.begin(), next.end(), [](Real value) { return value != 0; })) break;
                front.swap(next);
                Real weight = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    if (!front[i]) continue;
                    seen[i] = 1;
                    if (eligible[i] && !frozen_flag[i]) weight += gbar[i];
                }
                if (!(weight > 0)) continue;
                const Real share = remaining / weight;
                std::vector<Real> added(n, 0);
                remaining = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    if (!front[i] || !eligible[i] || frozen_flag[i]) continue;
                    const Real proposed = delta[i] + share;
                    const Real accepted = admissible(i, proposed);
                    added[i] = accepted == proposed ? share : accepted - delta[i];
                    remaining += gbar[i] * (share - added[i]);
                }
                protect_cut_classes(delta, added, [&](auto, Real rejected) { remaining += rejected; });
                for (std::size_t i = 0; i < n; ++i) delta[i] += added[i];
            }
            unplaced_volume += std::abs(remaining);
            if (remaining == 0) ++redistributed;
            else ++skipped;
        }
        result.degeneracy_frozen_dof_indices.clear();
        for (std::size_t i = 0; i < n; ++i) {
            if (frozen_flag[i] > Real{0.0}) {
                ++frozen;
                result.degeneracy_frozen_dof_indices.push_back(i);
            }
        }
        Real change = 0.0;
        Real correction = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const Real value = transported_level_set[i] + delta[i];
            candidate[i] = value;
            change = std::max(change, std::abs(value - iterate[i]));
            correction = std::max(correction, std::abs(delta[i]));
        }
        correction_volume = 0;
        for (std::size_t i = 0; i < n; ++i) {
            correction_volume += gbar[i] * std::abs(delta[i]);
        }
        // If the Picard map is no longer contracting, damp its update only
        // inside the same builder topology. Test the raw residual for convergence.
        if (iteration >= 4 && change >= Real{0.8} * prior_change) {
            for (std::size_t i = 0; i < n; ++i) {
                trial[i] = iterate[i] + Real{0.5} * (candidate[i] - iterate[i]);
            }
            bool same_class = true;
            for (std::size_t c = 0; c < cells.size() && same_class; ++c) {
                same_class = cutClass(cells[c], trial, isovalue, tolerance, minimum_fraction) == transported_cut_class[c];
            }
            if (allTrue(context, same_class)) {
                candidate.swap(trial);
            }
        }
        prior_change = change;
        iterate.swap(candidate);
        result.iterations = iteration + 1;
        if (change <= std::max(kGeometryRelativeTolerance * correction, rounding_floor)) {
            result.converged = true;
            break;
        }
    }
    InterfaceSums final_sums(n);
    if (!evaluate_endpoint(iterate, final_sums)) {
        return result;
    }
    result.current_interface_cells = static_cast<std::size_t>(current_cut_cells);
    result.sign_preserving_skipped_dofs = skipped;
    result.sign_preserving_redistributed_dofs = redistributed;
    result.degeneracy_frozen_dofs = frozen;

    Real previous_flux = 0.0;
    for (const auto value : previous.f) {
        previous_flux += value;
    }
    result.previous_interface_flux = previous_flux;
    result.current_interface_flux = current_flux;
    result.kinematic_volume_change = Real{0.5} * dt * (previous_flux + current_flux);

    for (std::size_t i = 0; i < n; ++i) {
        const Real delta = std::abs(iterate[i] - transported_level_set[i]);
        if (delta > Real{0.0}) {
            ++result.corrected_dofs;
        }
        result.max_abs_correction = std::max(result.max_abs_correction, delta);
    }

    // ---- exact sharp volumes for the report ---------------------------------
    const auto previous_volume = computeLevelSetCutCellVolume(
        mesh, level_set_dofs, volume_options, previous_level_set);
    const auto transported_volume = computeLevelSetCutCellVolume(
        mesh, level_set_dofs, volume_options, transported_level_set);
    const auto reconciled_volume = computeLevelSetCutCellVolume(
        mesh, level_set_dofs, volume_options,
        std::span<const Real>(iterate.data(), iterate.size()));
    if (!previous_volume.success || !transported_volume.success ||
        !reconciled_volume.success) {
        result.diagnostic = "level-set kinematic reconciliation could not measure the sharp volume";
        return result;
    }
    result.previous_negative_volume = previous_volume.negative_volume;
    result.transported_negative_volume = transported_volume.negative_volume;
    result.reconciled_negative_volume = reconciled_volume.negative_volume;
    result.transported_volume_error =
        (transported_volume.negative_volume - previous_volume.negative_volume) -
        Real{0.5} * dt * (previous_flux + result.transported_interface_flux);
    result.reconciled_volume_error =
        (reconciled_volume.negative_volume - previous_volume.negative_volume) -
        result.kinematic_volume_change;

    const Real volume_scale = std::abs(previous_volume.negative_volume) + std::abs(reconciled_volume.negative_volume)
                            + dt * (std::abs(previous_flux) + std::abs(current_flux)) + correction_volume;
    const Real solver_volume_tolerance = kGeometryRelativeTolerance * correction_volume
                                       + Real{128} * std::numeric_limits<Real>::epsilon() * volume_scale;
    result.volume_tolerance = std::abs(temporal_volume_remainder) + solver_volume_tolerance;
    if (!std::isfinite(result.reconciled_volume_error) || !std::isfinite(result.volume_tolerance) ||
        !std::isfinite(unplaced_volume)) {
        result.diagnostic = "level-set kinematic reconciliation produced a nonfinite volume budget";
        return result;
    }
    // Opposite infeasible fluxes on separate interfaces can sum to zero.
    // Their unsatisfied local volume constraints cannot cancel one another.
    result.volume_balance_satisfied = std::abs(result.reconciled_volume_error) <= result.volume_tolerance &&
                                      unplaced_volume <= solver_volume_tolerance;
    if (!result.converged || !result.volume_balance_satisfied) {
        result.retryable_failure = true;
        std::ostringstream message;
        message.precision(std::numeric_limits<Real>::max_digits10);
        message << (result.converged ? "kinematic volume target unattainable on the constrained interface"
                                    : "kinematic fixed-point iteration limit")
                << ", sharp volume defect=" << result.reconciled_volume_error
                << ", tolerance=" << result.volume_tolerance
                << ", unplaced constrained volume=" << unplaced_volume;
        result.diagnostic = message.str();
        return result;
    }
    *reconciled_level_set = std::move(iterate);
    result.applied = result.corrected_dofs > 0u;
    result.success = true;
    result.diagnostic = "reconciled";
    return result;
}

} // namespace

LevelSetKinematicReconciliationResult reconcileLevelSetWithKinematicFlux(
    const assembly::IMeshAccess& mesh, const dofs::DofHandler& level_set_dofs,
    const dofs::DofHandler& velocity_dofs, Real isovalue, Real tolerance, Real dt,
    std::span<const Real> previous_level_set, std::span<const Real> transported_level_set,
    std::span<const Real> previous_velocity, std::span<const Real> transported_velocity,
    std::vector<Real>& reconciled_level_set, const spaces::FunctionSpace* velocity_space)
{
    return integrateKinematicStep(mesh, level_set_dofs, velocity_dofs, isovalue, tolerance, dt,
        previous_level_set, transported_level_set, previous_velocity, transported_velocity,
        &reconciled_level_set, velocity_space);
}

LevelSetKinematicEndpoint measureLevelSetKinematicEndpoint(
    const assembly::IMeshAccess& mesh, const dofs::DofHandler& level_set_dofs,
    const dofs::DofHandler& velocity_dofs, const spaces::FunctionSpace& velocity_space,
    Real isovalue, Real tolerance, std::span<const Real> level_set, std::span<const Real> velocity)
{
    // Identical endpoints give zero transport rate. Only V(phi) and F(phi,w)
    // are requested, independent of a time step.
    auto measured = integrateKinematicStep(mesh, level_set_dofs, velocity_dofs, isovalue,
        tolerance, Real{1}, level_set, level_set, velocity, velocity, nullptr, &velocity_space);
    return {measured.success, measured.previous_negative_volume, measured.previous_interface_flux,
            std::move(measured.diagnostic)};
}

} // namespace svmp::FE::level_set
