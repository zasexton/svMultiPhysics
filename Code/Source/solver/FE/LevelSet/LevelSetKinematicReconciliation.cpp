#include "LevelSet/LevelSetKinematicReconciliation.h"

#include "Assembly/CutIntegrationContext.h"
#include "Basis/NodeOrderingConventions.h"
#include "Dofs/EntityDofMap.h"
#include "Elements/ReferenceElement.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
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
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
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
    // A finite field or geometry vector can have an unrepresentable square.
    const Real squared = dot(a, a);
    return std::isnormal(squared) ? std::sqrt(squared) : std::hypot(a[0], a[1], a[2]);
}

// One owned affine simplex with its level-set DOFs and nodal velocities.
// x holds the physical corner coordinates; xi the reference-element corner
// coordinates, in which the generated-interface builder cuts the cell.
struct CellTrace {
    std::array<Real, 4> g{};
    std::array<Real, 4> f{};
    std::array<Real, 4> m{};
    bool kinematically_exact{true};
};

struct SimplexCell {
    GlobalIndex id{};
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
    bool previous_has_surface{};
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

struct NodalKinematicRate {
    Real value{};
    Real difference_roundoff{};
    Real transported_roundoff{};
};

// Accumulate the interface integrals of one cell.  Returns true when the
// cell is cut by {phi = c}.
bool accumulateCell(const SimplexCell& cell,
                    std::span<const Real> phi,
                    std::span<const NodalKinematicRate> rate,
                    bool previous,
                    Real isovalue,
                    Real dt,
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
        Real t = interfaces::levelSetEdgeFraction(sa, sb);
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
        Real coefficient_roundoff = 0.0;
        const auto& velocity = previous ? cell.previous_velocity : cell.transported_velocity;
        for (int j = 0; j < n; ++j) {
            const auto lj = lambda[static_cast<std::size_t>(j)];
            advective += lj * dot(velocity[static_cast<std::size_t>(j)], gradient);
            const auto& node = rate[cell.dof[static_cast<std::size_t>(j)]];
            rate_value += lj * node.value;
            coefficient_roundoff += lj * node.difference_roundoff;
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
        // The P1 trace uses these barycentric weights at this point. Distant
        // coefficients cannot certify an error in its material derivative.
        const Real evaluation_roundoff = coefficient_roundoff / dt +
            Real{64} * std::numeric_limits<Real>::epsilon() * (std::abs(advective) + std::abs(rate_value));
        trace.kinematically_exact = trace.kinematically_exact &&
                                    std::abs(advective + rate_value) <= evaluation_roundoff;
        for (int i = 0; i < n; ++i) {
            const auto li = lambda[static_cast<std::size_t>(i)];
            const auto dof = cell.dof[static_cast<std::size_t>(i)];
            trace.g[static_cast<std::size_t>(i)] += scaled * li;
            trace.f[static_cast<std::size_t>(i)] += scaled * li * advective;
            trace.m[static_cast<std::size_t>(i)] += scaled * li * rate_value;
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
// builder and the level-set geometry snapshot see it: the interface
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
            // The entire zero plateau belongs to the closed negative phase.
            const bool positive = std::any_of(s.begin(), s.begin() + n, [](Real value) { return value > 0; });
            *cut_negative_fraction = positive ? Real{0} : Real{1};
        }
        return 3;
    }
    const auto fraction = [&](int a, int b) {
        const Real sa = s[static_cast<std::size_t>(a)];
        return std::clamp(interfaces::levelSetEdgeFraction(sa, s[static_cast<std::size_t>(b)]), Real{0.0}, Real{1.0});
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
    Real positive_fraction = 0.0;
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
        positive_fraction = n_neg == 1 ? Real{1.0} - corner : corner;
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
        positive_fraction = n_neg == 1 ? Real{1.0} - corner : corner;
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
        const auto& p = cell.xi[static_cast<std::size_t>(pos[0])];
        const auto& q = cell.xi[static_cast<std::size_t>(pos[1])];
        const Real positive = tetVolume(p, ap, bp, q) + tetVolume(ap, bp, q, aq) + tetVolume(bp, q, aq, bq);
        positive_fraction = volume > Real{0.0} ? positive / volume : Real{0.0};
    }
    if (cut_negative_fraction) {
        *cut_negative_fraction = negative_fraction;
    }
    int code = measure <= tolerance ? 0 : (measure <= std::sqrt(tolerance) ? 1 : 2);
    if (negative_fraction > Real{0.0} && negative_fraction < minimum_fraction) {
        code += 4;
    }
    if (positive_fraction > Real{0.0} && positive_fraction < minimum_fraction) {
        code += 8;
    }
    return code;
}

[[nodiscard]] bool hasCutSurface(int classification) noexcept
{
    // The low two bits encode collapsed, nearly tangent, regular and uncut.
    const int geometry = classification % 4;
    return geometry == 1 || geometry == 2;
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

// Gather every rank's values in rank order on every rank.
[[nodiscard]] std::vector<Real> gatherAll(const Collective& context, const std::vector<Real>& local)
{
#if FE_HAS_MPI
    if (context.active) {
        int size = 1;
        MPI_Comm_size(context.communicator, &size);
        const bool fits = local.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max());
        int count = fits ? static_cast<int>(local.size()) : -1;
        std::vector<int> counts(static_cast<std::size_t>(size));
        MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, context.communicator);
        std::vector<int> displacements(counts.size());
        long long total = 0;
        for (std::size_t r = 0; r < counts.size(); ++r) {
            if (counts[r] < 0 || total > std::numeric_limits<int>::max() - counts[r]) {
                throw std::overflow_error("level-set kinematic reconciliation exceeds the MPI count range");
            }
            displacements[r] = static_cast<int>(total);
            total += counts[r];
        }
        std::vector<Real> all(static_cast<std::size_t>(total));
        MPI_Allgatherv(local.data(), count, realType(), all.data(), counts.data(),
                       displacements.data(), realType(), context.communicator);
        return all;
    }
#else
    (void)context;
#endif
    return local;
}

// ---- wall-trace kinematics ---------------------------------------------------
// On an impermeable boundary facet (w . n = 0 at both time levels) the trace of
// phi obeys phi_t + w . grad_Gamma phi = 0 by itself, so the roots of a cut wall
// facet travel along the wall characteristic (s' = w_parallel on a 2D wall)
// independently of the interior interface.  Every vertex of a cut wall facet
// takes the previous level set at the foot of its backward wall characteristic
// over the step, followed exactly enough for the step-mean tangential velocity
// that a steady linear velocity reproduces its own characteristic.  One positive
// factor per connected wall trace, the ratio of the transported to the
// characteristic slopes across its roots, keeps the transported representation
// scale without moving a root.  These vertices then keep that value, and their share
// of the interior correction goes to the interior trace nodes.

// Linear pieces of the impermeable walls near the interface: per corner the
// position, the previous level set and the step-mean tangential velocity.
constexpr std::size_t kWallPieceValues = 7u;

struct WallPieces {
    int corners{0};
    std::vector<Real> data;

    [[nodiscard]] std::size_t size() const noexcept
    {
        return corners > 0 ? data.size() / (kWallPieceValues * static_cast<std::size_t>(corners)) : 0u;
    }
    [[nodiscard]] const Real* corner(std::size_t piece, int k) const noexcept
    {
        return &data[(piece * static_cast<std::size_t>(corners) + static_cast<std::size_t>(k)) * kWallPieceValues];
    }
    [[nodiscard]] Point position(std::size_t piece, int k) const noexcept
    {
        const Real* v = corner(piece, k);
        return {{v[0], v[1], v[2]}};
    }
    [[nodiscard]] Real diameter(std::size_t piece) const noexcept
    {
        Real d = 0;
        for (int a = 0; a < corners; ++a) {
            for (int b = a + 1; b < corners; ++b) {
                d = std::max(d, norm(sub(position(piece, a), position(piece, b))));
            }
        }
        return d;
    }
};

// Barycentric coordinates of the projection of p onto a piece; returns the
// distance by which p misses the piece.
[[nodiscard]] Real wallCoordinates(const WallPieces& wall, std::size_t piece, const Point& p,
                                   std::array<Real, 3>& lambda) noexcept
{
    const auto x0 = wall.position(piece, 0);
    const auto e1 = sub(wall.position(piece, 1), x0);
    const auto r = sub(p, x0);
    Point projection = x0;
    if (wall.corners == 2) {
        const Real t = dot(r, e1) / dot(e1, e1);
        lambda = {{Real{1} - t, t, Real{0}}};
        for (std::size_t k = 0; k < 3u; ++k) projection[k] += t * e1[k];
    } else {
        const auto e2 = sub(wall.position(piece, 2), x0);
        const Real a11 = dot(e1, e1);
        const Real a12 = dot(e1, e2);
        const Real a22 = dot(e2, e2);
        const Real det = a11 * a22 - a12 * a12;
        const Real l1 = (dot(r, e1) * a22 - dot(r, e2) * a12) / det;
        const Real l2 = (a11 * dot(r, e2) - a12 * dot(r, e1)) / det;
        lambda = {{Real{1} - l1 - l2, l1, l2}};
        for (std::size_t k = 0; k < 3u; ++k) projection[k] += l1 * e1[k] + l2 * e2[k];
    }
    Real outside = 0;
    for (int k = 0; k < wall.corners; ++k) outside = std::max(outside, -lambda[static_cast<std::size_t>(k)]);
    return outside * wall.diameter(piece) + norm(sub(p, projection));
}

// Previous level set at the foot of the backward wall characteristic through
// p, or nothing if the characteristic leaves the sampled impermeable wall.
[[nodiscard]] std::optional<Real> previousValueAtWallFoot(const WallPieces& wall,
                                                          std::span<const std::size_t> candidates,
                                                          Point p, Real dt, int substeps, Real reach)
{
    std::size_t piece = 0;
    std::array<Real, 3> lambda{};
    const auto locate = [&](const Point& q) {
        Real best = std::numeric_limits<Real>::infinity();
        for (const auto c : candidates) {
            std::array<Real, 3> l{};
            const Real miss = wallCoordinates(wall, c, q, l);
            if (miss < best) {
                best = miss;
                piece = c;
                lambda = l;
            }
        }
        return best <= reach;
    };
    const auto blend = [&](std::size_t offset) {
        Point v{};
        for (int k = 0; k < wall.corners; ++k) {
            const Real* c = wall.corner(piece, k);
            for (std::size_t d = 0; d < 3u; ++d) v[d] += lambda[static_cast<std::size_t>(k)] * c[offset + d];
        }
        return v;
    };
    const auto velocity = [&](const Point& q, Point& v) {
        if (!locate(q)) return false;
        v = blend(4u);
        return true;
    };
    const auto shifted = [](const Point& q, Real scale, const Point& v) {
        return Point{{q[0] - scale * v[0], q[1] - scale * v[1], q[2] - scale * v[2]}};
    };
    const Real h = substeps > 0 ? dt / static_cast<Real>(substeps) : Real{0};
    for (int step = 0; step < substeps; ++step) {
        Point k1{}, k2{}, k3{}, k4{};
        if (!velocity(p, k1) || !velocity(shifted(p, Real{0.5} * h, k1), k2) ||
            !velocity(shifted(p, Real{0.5} * h, k2), k3) || !velocity(shifted(p, h, k3), k4)) {
            return std::nullopt;
        }
        Point mean{};
        for (std::size_t d = 0; d < 3u; ++d) mean[d] = (k1[d] + Real{2} * (k2[d] + k3[d]) + k4[d]) / Real{6};
        if (mean == Point{}) continue;
        if (!locate(shifted(p, h, mean))) return std::nullopt;
        p = blend(0u);  // stay on the wall
    }
    if (!locate(p)) return std::nullopt;
    Real value = 0;
    for (int k = 0; k < wall.corners; ++k) value += lambda[static_cast<std::size_t>(k)] * wall.corner(piece, k)[3];
    return value;
}

// The wall-characteristic zero of a cut wall-facet edge a-b, at the fraction
// t from a; the correction keeps it there.
struct WallCrossing {
    std::size_t a{0u};
    std::size_t b{0u};
    Real t{0};
};

struct WallTrace {
    std::vector<char> node;   // the DOF is a vertex of a cut impermeable wall facet
    std::vector<Real> delta;  // its wall-characteristic change from the transported value
    std::vector<WallCrossing> crossings;
    std::size_t dofs{0u};
    std::size_t unresolved{0u};
};

// Local part of the wall trace: the impermeable boundary facets of the owned
// cells, sampled as linear pieces at the Lagrange points of the velocity
// order, and the vertices of the facets cut by the transported level set.
struct LocalWallTrace {
    std::vector<Real> pieces;
    std::vector<Real> vertices;  // dof, x, y, z per vertex of a cut facet
    std::vector<Real> edges;     // dof, dof, length of every facet edge that changes sign
    Real speed{0};
    Real diameter{0};
    std::array<Real, 6> box{{std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::infinity(),
                             std::numeric_limits<Real>::infinity(), -std::numeric_limits<Real>::infinity(),
                             -std::numeric_limits<Real>::infinity(), -std::numeric_limits<Real>::infinity()}};
};

void collectWallFacet(const SimplexCell& cell, const std::vector<LocalIndex>& face_nodes, int order,
                      Real isovalue, std::span<const Real> previous,
                      std::span<const Real> transported, LocalWallTrace& local)
{
    const int m = cell.corners - 1;
    if (face_nodes.size() != static_cast<std::size_t>(m)) {
        throw std::invalid_argument("found a boundary face that is not a simplex facet");
    }
    std::array<std::size_t, 3> f{};
    for (int k = 0; k < m; ++k) {
        const auto local_node = face_nodes[static_cast<std::size_t>(k)];
        if (local_node < 0 || local_node >= cell.corners) {
            throw std::invalid_argument("found an invalid boundary face");
        }
        f[static_cast<std::size_t>(k)] = static_cast<std::size_t>(local_node);
    }
    const auto e1 = sub(cell.x[f[1]], cell.x[f[0]]);
    Point normal = m == 2 ? Point{{-e1[1], e1[0], Real{0}}} : cross(e1, sub(cell.x[f[2]], cell.x[f[0]]));
    const Real normal_length = norm(normal);
    if (!(normal_length > Real{0}) || !std::isfinite(normal_length)) {
        throw std::invalid_argument("found a degenerate boundary face");
    }
    for (auto& value : normal) value /= normal_length;

    // Lattice of the velocity order on the facet and its linear pieces.
    std::vector<std::array<Real, 3>> mu;
    std::vector<std::array<std::size_t, 3>> pieces;
    if (m == 2) {
        for (int i = 0; i <= order; ++i) {
            const Real t = static_cast<Real>(i) / order;
            mu.push_back({{Real{1} - t, t, Real{0}}});
            if (i < order) pieces.push_back({{static_cast<std::size_t>(i), static_cast<std::size_t>(i + 1), 0u}});
        }
    } else {
        std::vector<std::vector<std::size_t>> index(static_cast<std::size_t>(order + 1));
        for (int i = 0; i <= order; ++i) {
            for (int j = 0; i + j <= order; ++j) {
                index[static_cast<std::size_t>(i)].push_back(mu.size());
                mu.push_back({{Real{1} - static_cast<Real>(i + j) / order, static_cast<Real>(i) / order,
                               static_cast<Real>(j) / order}});
            }
        }
        const auto at = [&](int i, int j) { return index[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]; };
        for (int i = 0; i < order; ++i) {
            for (int j = 0; i + j < order; ++j) {
                pieces.push_back({{at(i, j), at(i + 1, j), at(i, j + 1)}});
                if (i + j + 1 < order) pieces.push_back({{at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)}});
            }
        }
    }
    // Normal flow is relative to this facet's velocity trace at each time
    // level. A large velocity elsewhere cannot make an open facet impermeable.
    struct FacetVelocity {
        Real maximum_component{};
        Real maximum_normal_speed{};

        void sample(const Point& velocity, const Point& normal)
        {
            for (const Real value : velocity) {
                maximum_component = std::max(maximum_component, std::abs(value));
            }
            maximum_normal_speed = std::max(maximum_normal_speed, std::abs(dot(velocity, normal)));
        }

        [[nodiscard]] bool permeable() const
        {
            return maximum_normal_speed > std::sqrt(std::numeric_limits<Real>::epsilon()) * maximum_component;
        }
    };
    FacetVelocity previous_trace, transported_trace;
    Real facet_speed = 0;
    std::vector<std::array<Real, kWallPieceValues>> samples(mu.size());
    for (std::size_t q = 0; q < mu.size(); ++q) {
        Point position{}, reference{}, w_previous{}, w_transported{};
        Real phi = 0;
        for (int k = 0; k < m; ++k) {
            const Real weight = mu[q][static_cast<std::size_t>(k)];
            const auto corner = f[static_cast<std::size_t>(k)];
            phi += weight * previous[cell.dof[corner]];
            for (std::size_t d = 0; d < 3u; ++d) {
                position[d] += weight * cell.x[corner][d];
                reference[d] += weight * cell.xi[corner][d];
                w_previous[d] += weight * cell.previous_velocity[corner][d];
                w_transported[d] += weight * cell.transported_velocity[corner][d];
            }
        }
        if (cell.higher_order_velocity != nullptr) {
            const spaces::FunctionSpace::Value xi{reference[0], reference[1], reference[2]};
            const auto a = cell.higher_order_velocity->evaluate(xi, cell.previous_coefficients);
            const auto b = cell.higher_order_velocity->evaluate(xi, cell.transported_coefficients);
            w_previous = {{a[0], a[1], cell.corners == 4 ? a[2] : Real{0}}};
            w_transported = {{b[0], b[1], cell.corners == 4 ? b[2] : Real{0}}};
        }
        previous_trace.sample(w_previous, normal);
        transported_trace.sample(w_transported, normal);
        Point mean{};
        for (std::size_t d = 0; d < 3u; ++d) mean[d] = Real{0.5} * (w_previous[d] + w_transported[d]);
        const Real normal_part = dot(mean, normal);
        for (std::size_t d = 0; d < 3u; ++d) mean[d] -= normal_part * normal[d];
        facet_speed = std::max(facet_speed, norm(mean));
        samples[q] = {{position[0], position[1], position[2], phi, mean[0], mean[1], mean[2]}};
    }
    if (previous_trace.permeable() || transported_trace.permeable()) {
        return;
    }
    local.speed = std::max(local.speed, facet_speed);
    for (const auto& piece : pieces) {
        for (int k = 0; k < m; ++k) {
            const auto& sample = samples[piece[static_cast<std::size_t>(k)]];
            local.pieces.insert(local.pieces.end(), sample.begin(), sample.end());
        }
    }
    bool negative = false;
    bool other = false;
    for (int k = 0; k < m; ++k) {
        const auto corner = f[static_cast<std::size_t>(k)];
        (transported[cell.dof[corner]] - isovalue < Real{0} ? negative : other) = true;
        for (int l = k + 1; l < m; ++l) {
            local.diameter = std::max(local.diameter, norm(sub(cell.x[corner], cell.x[f[static_cast<std::size_t>(l)]])));
        }
    }
    if (!(negative && other)) {
        return;
    }
    for (int k = 0; k < m; ++k) {
        const auto corner = f[static_cast<std::size_t>(k)];
        const auto& x = cell.x[corner];
        local.vertices.insert(local.vertices.end(),
                              {static_cast<Real>(cell.dof[corner]), x[0], x[1], x[2]});
        for (std::size_t d = 0; d < 3u; ++d) {
            local.box[d] = std::min(local.box[d], x[d]);
            local.box[d + 3u] = std::max(local.box[d + 3u], x[d]);
        }
        for (int l = k + 1; l < m; ++l) {
            const auto other = f[static_cast<std::size_t>(l)];
            if ((transported[cell.dof[corner]] - isovalue < Real{0}) !=
                (transported[cell.dof[other]] - isovalue < Real{0})) {
                local.edges.insert(local.edges.end(), {static_cast<Real>(cell.dof[corner]),
                                                       static_cast<Real>(cell.dof[other]),
                                                       norm(sub(x, cell.x[other]))});
            }
        }
    }
}

// Replicated union-find over DOF numbers.
struct DofSets {
    std::map<std::size_t, std::size_t> parent;
    std::size_t find(std::size_t a)
    {
        auto [it, inserted] = parent.try_emplace(a, a);
        while (it->second != a) {
            a = it->second;
            it = parent.find(a);
        }
        return a;
    }
    void unite(std::size_t a, std::size_t b)
    {
        a = find(a);
        b = find(b);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
    }
};

// Collective: the wall-characteristic values of every vertex of a cut
// impermeable facet, identical on every rank.
[[nodiscard]] WallTrace wallTraceKinematics(const Collective& context, const LocalWallTrace& local, int corners,
                                            std::size_t n, Real isovalue, Real tolerance, Real dt,
                                            std::span<const Real> previous, std::span<const Real> transported)
{
    std::vector<Real> extremes{local.speed, local.diameter, -local.box[0], -local.box[1], -local.box[2],
                               local.box[3], local.box[4], local.box[5]};
    maxInPlace(context, extremes);
    const Real speed = extremes[0];
    const Real diameter = extremes[1];
    const Real radius = Real{1.5} * speed * dt + diameter;
    const std::size_t stride = kWallPieceValues * static_cast<std::size_t>(corners);
    const auto bounds = [&](const Real* piece, std::array<Real, 6>& box) {
        box = {{piece[0], piece[1], piece[2], piece[0], piece[1], piece[2]}};
        for (int k = 1; k < corners; ++k) {
            for (std::size_t d = 0; d < 3u; ++d) {
                box[d] = std::min(box[d], piece[static_cast<std::size_t>(k) * kWallPieceValues + d]);
                box[d + 3u] = std::max(box[d + 3u], piece[static_cast<std::size_t>(k) * kWallPieceValues + d]);
            }
        }
    };
    std::vector<Real> near_pieces;
    for (std::size_t p = 0; p + stride <= local.pieces.size(); p += stride) {
        std::array<Real, 6> box{};
        bounds(&local.pieces[p], box);
        bool meets = true;
        for (std::size_t d = 0; d < 3u; ++d) {
            meets = meets && box[d] <= extremes[d + 5u] + radius && box[d + 3u] >= -extremes[d + 2u] - radius;
        }
        if (meets) near_pieces.insert(near_pieces.end(), &local.pieces[p], &local.pieces[p] + stride);
    }
    const WallPieces wall{corners, gatherAll(context, near_pieces)};
    const auto vertices = gatherAll(context, local.vertices);
    const auto edges = gatherAll(context, local.edges);

    WallTrace trace;
    trace.node.assign(n, 0);
    trace.delta.assign(n, 0);
    std::map<std::size_t, Point> position;
    for (std::size_t v = 0; v + 4u <= vertices.size(); v += 4u) {
        position.try_emplace(static_cast<std::size_t>(vertices[v]),
                             Point{{vertices[v + 1u], vertices[v + 2u], vertices[v + 3u]}});
    }
    if (position.empty()) {
        return trace;
    }
    std::vector<std::array<Real, 6>> boxes(wall.size());
    Real smallest = std::numeric_limits<Real>::infinity();
    Real gradient = 0;
    for (std::size_t p = 0; p < wall.size(); ++p) {
        bounds(wall.corner(p, 0), boxes[p]);
        smallest = std::min(smallest, wall.diameter(p));
        for (int a = 0; a < corners; ++a) {
            for (int b = a + 1; b < corners; ++b) {
                Point dv{};
                for (std::size_t d = 0; d < 3u; ++d) dv[d] = wall.corner(p, a)[4u + d] - wall.corner(p, b)[4u + d];
                gradient = std::max(gradient, norm(dv) / norm(sub(wall.position(p, a), wall.position(p, b))));
            }
        }
    }
    // Classical Runge-Kutta substeps of at most 1/16 cell whose truncation
    // error, (dt |grad w| / N)^5 N / 120, stays at roundoff: a steady linear
    // velocity then reproduces its own characteristic.
    const Real truncation = std::pow(dt * gradient, Real{1.25}) /
                            std::pow(Real{120} * std::numeric_limits<Real>::epsilon(), Real{0.25});
    const int substeps = !(speed > Real{0}) ? 0
        : static_cast<int>(std::clamp(std::ceil(std::max(Real{16} * speed * dt / smallest, truncation)),
                                      Real{1}, Real{4096}));
    const Real reach = Real{1.0e-2} * diameter;
    std::map<std::size_t, Real> target;
    for (const auto& [dof, x] : position) {
        trace.node[dof] = 1;
        ++trace.dofs;
        if (signClass(transported[dof], isovalue, tolerance) == 0) {
            continue;  // a pinned interface vertex keeps its value
        }
        std::vector<std::size_t> candidates;
        for (std::size_t p = 0; p < wall.size(); ++p) {
            bool meets = true;
            for (std::size_t d = 0; d < 3u; ++d) {
                meets = meets && boxes[p][d] <= x[d] + radius && boxes[p][d + 3u] >= x[d] - radius;
            }
            if (meets) candidates.push_back(p);
        }
        const auto value = previousValueAtWallFoot(wall, candidates, x, dt, substeps, reach);
        if (value && std::isfinite(*value)) {
            target.emplace(dof, *value - isovalue);
        } else {
            ++trace.unresolved;
        }
    }
    // One factor per connected wall trace leaves its roots in place.  It is the
    // ratio of the transported to the characteristic slopes across the roots,
    // so a shifted trace keeps its values and a rescaled one its scale.
    DofSets sets;
    for (std::size_t e = 0; e + 3u <= edges.size(); e += 3u) {
        sets.unite(static_cast<std::size_t>(edges[e]), static_cast<std::size_t>(edges[e + 1u]));
    }
    std::map<std::size_t, std::array<Real, 2>> slopes;
    for (std::size_t e = 0; e + 3u <= edges.size(); e += 3u) {
        const auto a = static_cast<std::size_t>(edges[e]);
        const auto b = static_cast<std::size_t>(edges[e + 1u]);
        const auto ta = target.find(a);
        const auto tb = target.find(b);
        if (ta == target.end() || tb == target.end()) continue;
        auto& sums = slopes[sets.find(a)];
        sums[0] += std::abs(transported[a] - transported[b]) / edges[e + 2u];
        sums[1] += std::abs(ta->second - tb->second) / edges[e + 2u];
    }
    for (const auto& [dof, s] : target) {
        const auto found = slopes.find(sets.find(dof));
        const Real scale = found != slopes.end() ? found->second[0] / found->second[1] : Real{1};
        const Real factor = scale > Real{0} && std::isfinite(scale) ? scale : Real{1};
        trace.delta[dof] = factor * s - (transported[dof] - isovalue);
    }
    // Every cut facet edge whose two wall values straddle the isovalue fixes
    // its zero; the vertex values stay free otherwise. An unresolved vertex
    // keeps its transported value as its wall value; a pinned one has none.
    // The zero is the root along the edge of the previous level set at the
    // characteristic foot, not the zero of the two vertex values: that one lags
    // behind a slope change at a vertex (the O(h^2/dt) re-interpolation error
    // of a semi-Lagrangian step) and stalls a slowly advancing trace.  Illinois
    // regula falsi starts from the vertex-value zero s and keeps it when it is
    // already the root.
    const auto edge_root = [&](std::size_t a, std::size_t b, Real g0, Real g1, Real s) -> std::optional<Real> {
        const Point& xa = position.at(a);
        const Point& xb = position.at(b);
        std::vector<std::size_t> candidates;
        for (std::size_t p = 0; p < wall.size(); ++p) {
            bool meets = true;
            for (std::size_t d = 0; d < 3u; ++d) {
                meets = meets && boxes[p][d] <= std::max(xa[d], xb[d]) + radius &&
                        boxes[p][d + 3u] >= std::min(xa[d], xb[d]) - radius;
            }
            if (meets) candidates.push_back(p);
        }
        const Real small = Real{16} * std::numeric_limits<Real>::epsilon() * (std::abs(g0) + std::abs(g1));
        Real s0 = 0, s1 = 1;
        int kept = 0;
        for (int it = 0; it < 64; ++it) {
            Point x{};
            for (std::size_t d = 0; d < 3u; ++d) x[d] = xa[d] + s * (xb[d] - xa[d]);
            const auto value = previousValueAtWallFoot(wall, candidates, x, dt, substeps, reach);
            if (!value || !std::isfinite(*value)) return std::nullopt;
            const Real g = *value - isovalue;
            if (std::abs(g) <= small) return s;
            if ((g < Real{0}) == (g0 < Real{0})) {
                s0 = s;
                g0 = g;
                if (kept == 1) g1 *= Real{0.5};
                kept = 1;
            } else {
                s1 = s;
                g1 = g;
                if (kept == -1) g0 *= Real{0.5};
                kept = -1;
            }
            if (!(s1 - s0 > Real{4} * std::numeric_limits<Real>::epsilon())) break;
            s = (s0 * g1 - s1 * g0) / (g1 - g0);
        }
        return Real{0.5} * (s0 + s1);
    };
    std::set<std::pair<std::size_t, std::size_t>> seen_edges;
    const auto has_wall_value = [&](std::size_t dof) {
        return position.count(dof) != 0 && signClass(transported[dof], isovalue, tolerance) != 0;
    };
    for (std::size_t e = 0; e + 3u <= edges.size(); e += 3u) {
        auto a = static_cast<std::size_t>(edges[e]);
        auto b = static_cast<std::size_t>(edges[e + 1u]);
        if (a > b) std::swap(a, b);
        if (!has_wall_value(a) || !has_wall_value(b) || !seen_edges.insert({a, b}).second) continue;
        const Real va = transported[a] - isovalue + trace.delta[a];
        const Real vb = transported[b] - isovalue + trace.delta[b];
        if (!(va * vb < Real{0})) continue;
        Real t = interfaces::levelSetEdgeFraction(va, vb);
        const auto ta = target.find(a);
        const auto tb = target.find(b);
        if (ta != target.end() && tb != target.end() && ta->second * tb->second < Real{0}) {
            if (const auto root = edge_root(a, b, ta->second, tb->second, t)) t = *root;
        }
        trace.crossings.push_back({a, b, t});
    }
    return trace;
}

// Connected interface components: the edges of a supported cell that change
// sign in either state belong together, and so do all cells around a node
// that the interface swept over during the step (its sign changed).  A node
// on cut edges of two components is shared by them.  Labels are dense and
// identical on every rank.
struct InterfaceComponents {
    std::vector<long long> node;   // component of each DOF (the smallest if shared), -1 if none
    std::vector<char> shared;      // the DOF lies on cut edges of two or more components
    std::vector<long long> cell;   // component of each owned cell, -1 without a cut edge
    std::map<std::pair<std::size_t, long long>, std::size_t> splits;  // (shared DOF, component) -> index
    std::size_t count{0};
};

[[nodiscard]] InterfaceComponents interfaceComponents(const Collective& context,
                                                      const std::vector<SimplexCell>& cells,
                                                      std::span<const Real> previous,
                                                      std::span<const Real> transported,
                                                      Real isovalue, std::size_t n)
{
    const auto negative = [&](std::span<const Real> phi, std::size_t i) { return phi[i] - isovalue < 0; };
    std::vector<Real> local;
    std::vector<std::pair<std::size_t, std::size_t>> first_edge(cells.size(), {n, n});
    for (std::size_t c = 0; c < cells.size(); ++c) {
        const auto& cell = cells[c];
        std::vector<std::size_t> edges;
        for (int a = 0; a < cell.corners; ++a) {
            for (int b = a + 1; b < cell.corners; ++b) {
                const auto i = cell.dof[static_cast<std::size_t>(a)];
                const auto j = cell.dof[static_cast<std::size_t>(b)];
                if (negative(previous, i) != negative(previous, j) ||
                    negative(transported, i) != negative(transported, j)) {
                    edges.push_back(std::min(i, j));
                    edges.push_back(std::max(i, j));
                }
            }
        }
        if (edges.empty()) continue;
        first_edge[c] = {edges[0], edges[1]};
        for (int a = 0; a < cell.corners; ++a) {
            const auto i = cell.dof[static_cast<std::size_t>(a)];
            if (negative(previous, i) != negative(transported, i)) {
                edges.push_back(i);
                edges.push_back(i);
            }
        }
        local.push_back(static_cast<Real>(edges.size() / 2u));
        for (const auto dof : edges) local.push_back(static_cast<Real>(dof));
    }
    const auto all = gatherAll(context, local);
    // Union-find over edges; a swept node is the pseudo-edge (i, i).
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> edge_id;
    std::vector<std::size_t> parent;
    const auto find = [&](std::size_t e) {
        while (parent[e] != e) e = parent[e] = parent[parent[e]];
        return e;
    };
    for (std::size_t r = 0; r < all.size();) {
        const auto count = static_cast<std::size_t>(all[r++]);
        std::size_t first = 0;
        for (std::size_t e = 0; e < count; ++e, r += 2u) {
            const auto key = std::make_pair(static_cast<std::size_t>(all[r]), static_cast<std::size_t>(all[r + 1u]));
            const auto [it, inserted] = edge_id.try_emplace(key, parent.size());
            if (inserted) parent.push_back(parent.size());
            if (e == 0) {
                first = it->second;
            } else {
                const auto a = find(first);
                const auto b = find(it->second);
                if (a != b) parent[std::max(a, b)] = std::min(a, b);
            }
        }
    }
    InterfaceComponents components;
    std::map<std::size_t, long long> dense;
    for (const auto& [key, id] : edge_id) dense.emplace(find(id), 0);
    for (auto& [root, label] : dense) label = static_cast<long long>(components.count++);
    components.node.assign(n, -1);
    components.shared.assign(n, 0);
    std::set<std::pair<std::size_t, long long>> memberships;
    for (const auto& [key, id] : edge_id) {
        const auto label = dense[find(id)];
        for (const auto dof : {key.first, key.second}) {
            memberships.emplace(dof, label);
            auto& node = components.node[dof];
            if (node >= 0 && node != label) components.shared[dof] = 1;
            node = node < 0 ? label : std::min(node, label);
        }
    }
    for (const auto& membership : memberships) {
        if (components.shared[membership.first]) {
            components.splits.emplace(membership, components.splits.size());
        }
    }
    components.cell.assign(cells.size(), -1);
    for (std::size_t c = 0; c < cells.size(); ++c) {
        if (first_edge[c].first < n) components.cell[c] = dense[find(edge_id.at(first_edge[c]))];
    }
    return components;
}

// A read-only endpoint measurement stops after the first trace integration.
// A supplied flag enables the connected-patch search for a shared node and
// records whether any of its volume was distributed beyond the first ring.
// The first attempt can measure a candidate at field-scale roundoff; when that
// fails its volume tests, a second attempt can resolve changes at each node.
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
    const spaces::FunctionSpace* velocity_space,
    bool* used_connected_shared_patch = nullptr,
    bool* stopped_above_nodal_roundoff = nullptr)
{
    LevelSetKinematicReconciliationResult result;
    const auto context = collectiveFor(level_set_dofs);
    const auto n = static_cast<std::size_t>(level_set_dofs.getNumDofs());
    const auto n_velocity = static_cast<std::size_t>(velocity_dofs.getNumDofs());
    const int dimension = mesh.dimension();

    const Real minimum_fraction = assembly::CutIntegrationContext::minGeneratedCutVolumeFraction();
    const Real geometry_tolerance = tolerance > 0 ? tolerance : LevelSetVolumeOptions{}.tolerance;
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
    const auto original_transported = transported_level_set;
    std::vector<Real> previous_geometry, transported_geometry;
    if (local_failure.empty()) {
        try {
            // Use one represented geometry for the traces, wall intersections,
            // component support, rates and sharp volumes. Band vertices stay fixed.
            const auto canonicalize = [&](std::span<const Real>& field, std::vector<Real>& storage) {
                const auto in_band = [&](Real value) {
                    return value != isovalue && std::abs(value - isovalue) <= tolerance;
                };
                if (std::any_of(field.begin(), field.end(), in_band)) {
                    storage.assign(field.begin(), field.end());
                    for (auto& value : storage) {
                        if (in_band(value)) {
                            value = isovalue;
                        }
                    }
                    field = storage;
                }
            };
            const bool same_field = previous_level_set.data() == transported_level_set.data();
            canonicalize(previous_level_set, previous_geometry);
            if (same_field) {
                transported_level_set = previous_level_set;
            } else {
                canonicalize(transported_level_set, transported_geometry);
            }
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
                cell.id = cell_id;
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
                cell.previous_has_surface = hasCutSurface(cutClass(
                    cell, previous_level_set, isovalue, geometry_tolerance, minimum_fraction, &cell.previous_fraction));
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
    constexpr Real roundoff_factor = Real{64} * std::numeric_limits<Real>::epsilon();
    std::vector<NodalKinematicRate> rate(n);
    std::vector<Real> field_roundoff(1, 0);
    for (std::size_t i = 0; i < n; ++i) {
        auto& node = rate[i];
        node.value = (transported_level_set[i] - previous_level_set[i]) / dt;
        node.transported_roundoff = roundoff_factor * std::abs(transported_level_set[i] - isovalue);
        const Real previous_roundoff = roundoff_factor * std::abs(previous_level_set[i] - isovalue);
        node.difference_roundoff = std::max(previous_roundoff, node.transported_roundoff);
        field_roundoff.front() = std::max(field_roundoff.front(), node.transported_roundoff);
    }
    maxInPlace(context, field_roundoff);
    InterfaceSums previous(n);
    unsigned long long previous_cut_cells = 0u;
    try {
        for (auto& cell : cells) {
            if (cell.previous_has_surface && accumulateCell(cell, previous_level_set, rate,
                               true, isovalue, dt, previous, cell.previous_trace)) {
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
    } else {
        volume_options.coefficient_classification_policy =
            interfaces::LevelSetCoefficientClassificationPolicy::ExactRepresentedZero;
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

    // ---- wall-trace kinematics (collective) ----------------------------------
    LocalWallTrace local_wall;
    try {
        std::unordered_map<GlobalIndex, std::size_t> owned;
        for (std::size_t c = 0; c < cells.size(); ++c) owned.emplace(cells[c].id, c);
        const auto& reference = elements::ReferenceElement::shared(
            dimension == 2 ? ElementType::Triangle3 : ElementType::Tetra4);
        mesh.forEachBoundaryFace(-1, [&](GlobalIndex face, GlobalIndex cell_id) {
            const auto found = owned.find(cell_id);
            if (found == owned.end()) return;
            const auto& cell = cells[found->second];
            const auto local_face = mesh.getLocalFaceIndex(face, cell_id);
            if (local_face < 0 || static_cast<std::size_t>(local_face) >= reference.num_faces()) {
                throw std::invalid_argument("found a boundary face outside its cell");
            }
            const int order = cell.higher_order_velocity ? std::max(1, velocity_space->polynomial_order(cell_id)) : 1;
            collectWallFacet(cell, reference.face_nodes(static_cast<std::size_t>(local_face)), order, isovalue,
                             previous_level_set, transported_level_set, local_wall);
        });
    } catch (const std::exception& error) {
        local_failure = error.what();
    }
    if (!allTrue(context, local_failure.empty())) {
        result.diagnostic = local_failure.empty() ? "another rank failed to sample the impermeable walls"
                                                  : "level-set kinematic reconciliation " + local_failure;
        return result;
    }
    const auto wall = wallTraceKinematics(context, local_wall, dimension, n, isovalue, tolerance, dt,
                                          previous_level_set, transported_level_set);
    const auto component = interfaceComponents(context, cells, previous_level_set, transported_level_set, isovalue, n);
    result.wall_trace_dofs = wall.dofs;
    result.wall_trace_unresolved_dofs = wall.unresolved;

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
            cells[c], transported_level_set, isovalue, geometry_tolerance, minimum_fraction);
    }
    std::size_t skipped = 0u;
    std::size_t redistributed = 0u;
    std::size_t frozen = 0u;
    // For each simplex, account locally for the difference between its exact
    // swept volume and the trapezoidal shape derivative. Distribute the scalar
    // defect to the same nodal interface tests by their integrated weights.
    // An uncorrected, exactly advected trace keeps its legitimate temporal
    // quadrature remainder. A node shared by two interface components gets
    // its residual and weight separately for each (the splits), and the final
    // evaluation sums the budget of every component.
    std::vector<bool> supported_cell(cells.size());
    Real temporal_volume_remainder = 0;
    const std::size_t splits = component.splits.size();
    std::vector<Real> split_g(splits, 0), split_r(splits, 0);
    // Per component: volume change, previous and current flux, exact remainder, volume scale.
    constexpr std::size_t kBudgetTerms = 5u;
    std::vector<Real> budget;
    auto evaluate_endpoint = [&](std::span<const Real> endpoint, InterfaceSums& current, bool with_budget) {
        unsigned long long local_cut_cells = 0;
        std::vector<Real> remainder(1, 0);
        std::vector<Real> split_terms(2u * splits, 0);
        budget.assign(with_budget ? kBudgetTerms * component.count : 0u, 0);
        try {
            for (std::size_t c = 0; c < cells.size(); ++c) {
                const auto& cell = cells[c];
                CellTrace trace;
                Real fraction = 0;
                const auto classification = cutClass(cell, endpoint, isovalue, geometry_tolerance, minimum_fraction, &fraction);
                if (hasCutSurface(classification) &&
                    accumulateCell(cell, endpoint, rate, false, isovalue, dt, current, trace)) {
                    ++local_cut_cells;
                }
                Real defect = cell.measure * (fraction - cell.previous_fraction);
                Real weight = 0;
                bool corrected = false;
                for (int j = 0; j < cell.corners; ++j) {
                    const Real g = Real{0.5} * (cell.previous_trace.g[j] + trace.g[j]);
                    defect += g * (endpoint[cell.dof[j]] - previous_level_set[cell.dof[j]]);
                    weight += g;
                    const auto dof = cell.dof[j];
                    corrected = corrected ||
                                std::abs(endpoint[dof] - transported_level_set[dof]) > rate[dof].transported_roundoff;
                }
                supported_cell[c] = weight > 0;
                const bool exact = !corrected && trace.kinematically_exact &&
                                   cell.previous_trace.kinematically_exact;
                if (exact && weight > 0) {
                    remainder[0] += defect;
                } else if (weight > 0) {
                    for (int j = 0; j < cell.corners; ++j) {
                        const Real share = defect * (cell.previous_trace.g[j] + trace.g[j]) / (dt * weight);
                        current.m[cell.dof[j]] -= share;
                        trace.m[j] -= share;
                    }
                }
                const auto label = component.cell[c];
                if (label < 0) continue;
                for (int j = 0; j < cell.corners; ++j) {
                    if (!component.shared[cell.dof[j]]) continue;
                    const auto k = component.splits.at({cell.dof[j], label});
                    split_terms[k] += Real{0.5} * (cell.previous_trace.g[j] + trace.g[j]);
                    split_terms[splits + k] += Real{0.5} * (cell.previous_trace.m[j] + trace.m[j] +
                                                            cell.previous_trace.f[j] + trace.f[j]);
                }
                if (with_budget) {
                    Real* terms = &budget[kBudgetTerms * static_cast<std::size_t>(label)];
                    terms[0] += cell.measure * (fraction - cell.previous_fraction);
                    for (int j = 0; j < cell.corners; ++j) {
                        terms[1] += cell.previous_trace.f[j];
                        terms[2] += trace.f[j];
                    }
                    terms[3] += exact && weight > 0 ? defect : Real{0};
                    terms[4] += cell.measure * (fraction + cell.previous_fraction);
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
        sumInPlace(context, split_terms);
        std::copy(split_terms.begin(), split_terms.begin() + static_cast<std::ptrdiff_t>(splits), split_g.begin());
        std::copy(split_terms.begin() + static_cast<std::ptrdiff_t>(splits), split_terms.end(), split_r.begin());
        sumInPlace(context, budget);
        current.reduce(context);
        current_cut_cells = sumCount(context, local_cut_cells);
        current_flux = 0.0;
        for (const auto value : current.f) {
            current_flux += value;
        }

        return true;
    };
    // A node may not change its sign class and may not move toward the
    // isovalue by more than half of its transported distance: the cut points
    // of its edges then keep their order and stay away from the vertex, so
    // neither the cut topology nor its degeneracy class changes.
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
    // A wall-trace node takes its wall-characteristic value within the same limits.
    std::vector<char> wall_node = wall.node;
    std::vector<Real> wall_delta(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        if (wall_node[i] && transported_class[i] != 0) {
            wall_delta[i] = admissible(i, wall.delta[i]);
            result.wall_trace_limited_dofs += wall_delta[i] != wall.delta[i] ? 1u : 0u;
        }
    }
    const auto previous_volume = computeLevelSetCutCellVolume(
        mesh, level_set_dofs, volume_options, previous_level_set);
    Real previous_flux = 0.0;
    for (const auto value : previous.f) {
        previous_flux += value;
    }
    const auto solver_tolerance = [&](Real reconciled_volume, Real correction_volume) {
        const Real volume_scale = std::abs(previous_volume.negative_volume) + std::abs(reconciled_volume)
                                + dt * (std::abs(previous_flux) + std::abs(current_flux)) + correction_volume;
        return kGeometryRelativeTolerance * correction_volume
             + Real{128} * std::numeric_limits<Real>::epsilon() * volume_scale;
    };
    Real prior_change = std::numeric_limits<Real>::infinity();
    Real correction_volume = 0;
    Real unplaced_volume = 0;
    std::map<long long, Real> unplaced_by_interface;
    std::vector<char> limited(n, 0);
    std::set<std::size_t> projected_groups;
    for (int iteration = 0; iteration < kMaxGeometryIterations; ++iteration) {
        InterfaceSums current(n);
        if (!evaluate_endpoint(iterate, current, false)) {
            return result;
        }
        if (iteration == 0) {
            result.transported_interface_flux = current_flux;
        }
        // Lumped correction of every node that carries interface measure.
        // The limited part of its correction is moved to the available tests
        // on its connected cut-cell patch.
        unplaced_volume = 0;
        skipped = 0u;
        redistributed = 0u;
        std::vector<Real> delta(n, 0.0);
        std::vector<Real> gbar(n, 0.0);
        std::vector<Real> deficit(n, 0.0);
        std::vector<char> eligible(n, 0);
        std::vector<Real> free_step(n, std::numeric_limits<Real>::quiet_NaN());
        bool any_limited = false;
        for (std::size_t i = 0; i < n; ++i) {
            gbar[i] = Real{0.5} * (previous.g[i] + current.g[i]);
            delta[i] = wall_delta[i];
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
            free_step[i] = step;
            if (component.shared[i]) {
                continue;  // its share is split by component below
            }
            if (transported_class[i] == 0 || wall_node[i]) {
                // A pinned interface vertex or a wall-trace node carries flux
                // even though its coefficient follows another law. Its other
                // trace nodes inherit the rest of its share.
                deficit[i] = gbar[i] * (step - delta[i]);
                any_limited = any_limited || deficit[i] != 0;
                continue;
            }
            // A node once limited stops receiving shares, so that its
            // change stays continuous in the geometry and the iteration
            // cannot cycle between two active sets.
            delta[i] = admissible(i, step);
            if (delta[i] == step && !limited[i]) {
                eligible[i] = 1;
            } else {
                limited[i] = 1;
                deficit[i] = gbar[i] * (step - delta[i]);
                any_limited = true;
            }
        }
        // A connected group of cut wall-facet edges keeps the zero of every
        // edge at its wall-characteristic fraction t, while its vertex values
        // follow the lumped steps s as far as those zeros allow: the
        // gbar-weighted least-squares projection of s onto
        //     (1 - t) (phi_a + delta_a - c) + t (phi_b + delta_b - c) = 0.
        // The rest of their share goes to the other trace nodes. A group that
        // cannot be projected admissibly keeps its wall-characteristic values.
        {
            DofSets groups;
            std::vector<const WallCrossing*> active;
            for (const auto& crossing : wall.crossings) {
                if (!wall_node[crossing.a] || !wall_node[crossing.b]) continue;
                active.push_back(&crossing);
                groups.unite(crossing.a, crossing.b);
            }
            std::map<std::size_t, std::vector<const WallCrossing*>> by_group;
            for (const auto* crossing : active) by_group[groups.find(crossing->a)].push_back(crossing);
            struct Row {
                std::vector<std::pair<std::size_t, Real>> c;
                Real rhs{0};
            };
            for (const auto& [root, crossings] : by_group) {
                std::vector<std::size_t> nodes;
                for (const auto* crossing : crossings) {
                    nodes.push_back(crossing->a);
                    nodes.push_back(crossing->b);
                }
                std::sort(nodes.begin(), nodes.end());
                nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
                bool usable = true;
                for (const auto i : nodes) {
                    usable = usable && gbar[i] > Real{0} && std::isfinite(free_step[i]) &&
                             !component.shared[i] && transported_class[i] != 0;
                }
                if (!usable) continue;
                // Relative to the wall values w: delta = w + e with e the
                // projection of r = s - w.  The characteristic values v (before
                // the admissibility limit) need not have their zero at t.
                const auto r = [&](std::size_t i) { return free_step[i] - wall_delta[i]; };
                const auto v = [&](std::size_t i) {
                    return transported_level_set[i] - isovalue + wall.delta[i];
                };
                Real largest = 0;
                for (const auto i : nodes) largest = std::max(largest, std::abs(v(i)));
                std::vector<Real> miss;
                bool moved = false;
                for (const auto* x : crossings) {
                    miss.push_back((Real{1} - x->t) * v(x->a) + x->t * v(x->b));
                    moved = moved || std::abs(miss.back()) > Real{128} * std::numeric_limits<Real>::epsilon() * largest;
                }
                std::vector<Row> rows;
                for (std::size_t k = 0; k < crossings.size(); ++k) {
                    const auto* x = crossings[k];
                    rows.push_back({{{x->a, Real{1} - x->t}, {x->b, x->t}},
                                    -((moved ? miss[k] : Real{0}) + (Real{1} - x->t) * r(x->a) + x->t * r(x->b))});
                }
                // A group whose zeros already lie at t and whose own share cancels
                // within the solver tolerance keeps its wall values, so an exactly
                // transported trace stays exact; a group once projected stays
                // projected, so the set cannot cycle.
                if (!projected_groups.count(root)) {
                    Real share = 0;
                    Real magnitude = 0;
                    Real scale = 0;
                    for (const auto i : nodes) {
                        share += gbar[i] * r(i);
                        magnitude += gbar[i] * std::abs(r(i));
                        scale += gbar[i] * std::abs(transported_level_set[i] - isovalue);
                    }
                    if (!moved && !(std::abs(share) > kGeometryRelativeTolerance * magnitude +
                                                Real{128} * std::numeric_limits<Real>::epsilon() * scale)) {
                        continue;
                    }
                }
                // Normal equations (C W^-1 C^T) lambda = rhs, W = diag(gbar).
                const std::size_t m = rows.size();
                std::vector<Real> normal(m * m, 0), lambda(m, 0);
                for (std::size_t e = 0; e < m; ++e) {
                    lambda[e] = rows[e].rhs;
                    for (std::size_t f = 0; f < m; ++f) {
                        Real value = 0;
                        for (const auto& [i, ce] : rows[e].c) {
                            for (const auto& [j, cf] : rows[f].c) {
                                if (i == j) value += ce * cf / gbar[i];
                            }
                        }
                        normal[e * m + f] = value;
                    }
                }
                // Gaussian elimination with partial pivoting, scaled by row.
                bool solved = true;
                for (std::size_t k = 0; k < m && solved; ++k) {
                    std::size_t pivot = k;
                    for (std::size_t r = k + 1; r < m; ++r) {
                        if (std::abs(normal[r * m + k]) > std::abs(normal[pivot * m + k])) pivot = r;
                    }
                    Real row_scale = 0;
                    for (std::size_t c = 0; c < m; ++c) row_scale = std::max(row_scale, std::abs(normal[pivot * m + c]));
                    if (!(std::abs(normal[pivot * m + k]) > Real{64} * std::numeric_limits<Real>::epsilon() * row_scale)) {
                        solved = false;
                        break;
                    }
                    if (pivot != k) {
                        for (std::size_t c = 0; c < m; ++c) std::swap(normal[k * m + c], normal[pivot * m + c]);
                        std::swap(lambda[k], lambda[pivot]);
                    }
                    for (std::size_t r = k + 1; r < m; ++r) {
                        const Real factor = normal[r * m + k] / normal[k * m + k];
                        for (std::size_t c = k; c < m; ++c) normal[r * m + c] -= factor * normal[k * m + c];
                        lambda[r] -= factor * lambda[k];
                    }
                }
                if (!solved) continue;
                for (std::size_t k = m; k-- > 0;) {
                    for (std::size_t c = k + 1; c < m; ++c) lambda[k] -= normal[k * m + c] * lambda[c];
                    lambda[k] /= normal[k * m + k];
                }
                std::map<std::size_t, Real> projected;
                for (const auto i : nodes) projected[i] = wall_delta[i] + r(i);
                for (std::size_t e = 0; e < m; ++e) {
                    for (const auto& [i, ce] : rows[e].c) projected[i] += ce * lambda[e] / gbar[i];
                }
                bool admissible_group = true;
                for (const auto& [i, value] : projected) {
                    admissible_group = admissible_group && std::isfinite(value) && admissible(i, value) == value;
                }
                if (!admissible_group) continue;
                projected_groups.insert(root);
                for (const auto& [i, value] : projected) {
                    delta[i] = value;
                    deficit[i] = gbar[i] * (free_step[i] - value);
                    any_limited = any_limited || deficit[i] != 0;
                }
            }
        }
        // A node shared by interface components keeps its value (or its wall
        // value); the part of its share that belongs to each component stays
        // in that component, so no correction moves volume between them.
        std::vector<Real> split_deficit(splits, 0), split_dropped(splits, 0);
        for (const auto& [key, k] : component.splits) {
            if (!(gbar[key.first] > Real{0.0})) continue;
            split_deficit[k] = -dt * split_r[k] - split_g[k] * delta[key.first];
            any_limited = any_limited || split_deficit[k] != 0;
        }
        // Scatter a blocked test's signed volume to the other active tests
        // on its cut cells. Count shared cells as in the lumped one-ring
        // projection. A share with no local recipient is retained for the
        // wider connected patch below.
        std::vector<Real> frozen_flag(n, 0);
        std::vector<Real> unplaced(n, 0);
        std::vector<Real> split_unplaced(splits, 0);
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
        // The same for each component's part of a shared node, on its cells only.
        const auto scatter_split = [&](const std::vector<Real>& shares, std::vector<Real>& added) {
            if (splits == 0) return;
            std::vector<Real> patch(splits, 0), extra(n, 0);
            const auto visit_receivers = [&](const auto& add) {
                for (std::size_t c = 0; c < cells.size(); ++c) {
                    const auto label = component.cell[c];
                    if (!supported_cell[c] || label < 0) continue;
                    const auto& cell = cells[c];
                    for (int a = 0; a < cell.corners; ++a) {
                        if (!component.shared[cell.dof[a]]) continue;
                        const auto k = component.splits.at({cell.dof[a], label});
                        if (shares[k] == 0) continue;
                        for (int b = 0; b < cell.corners; ++b) {
                            const auto j = cell.dof[b];
                            if (eligible[j] && !frozen_flag[j]) add(k, j);
                        }
                    }
                }
            };
            visit_receivers([&](std::size_t k, std::size_t j) { patch[k] += gbar[j]; });
            sumInPlace(context, patch);
            visit_receivers([&](std::size_t k, std::size_t j) { extra[j] += shares[k] / patch[k]; });
            sumInPlace(context, extra);
            for (const auto& [key, k] : component.splits) {
                if (shares[k] == 0) continue;
                if (patch[k] > 0) ++redistributed;
                else split_unplaced[k] += shares[k];
            }
            for (std::size_t j = 0; j < n; ++j) added[j] += extra[j];
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
            scatter_split(split_deficit, added);
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
                    if (!touched || cutClass(cell, trial, isovalue, geometry_tolerance, minimum_fraction) ==
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
            if (component.shared[i]) {
                for (auto it = component.splits.lower_bound({i, -1}); it != component.splits.end() &&
                                                                       it->first.first == i; ++it) {
                    split_dropped[it->second] += share * split_g[it->second] / gbar[i];
                }
            } else {
                dropped[i] += share;
            }
            frozen_flag[i] = 1;
            any_frozen = true;
        });
        if (any_frozen) {
            auto added = scatter(dropped);
            scatter_split(split_dropped, added);
            limit_addition(added);
            protect_cut_classes(delta, added, [&](auto i, Real share) { unplaced[i] += share; });
            for (std::size_t i = 0; i < n; ++i) delta[i] += added[i];
        }

        // A frozen sliver can surround one of its vertices with other frozen
        // vertices. Carry its unplaced volume outward, one cut-cell ring at a
        // time, until it reaches an admissible trace. Never cross an uncut cell
        // or change the existing correction of a frozen node. What remains is
        // summed over its connected interface: opposite parts of one interface
        // cancel, those of separate interfaces cannot.
        unplaced_by_interface.clear();
        const auto distribute_remaining = [&](std::size_t source, long long label, Real remaining) {
            std::vector<Real> seen(n, 0), front(n, 0);
            seen[source] = front[source] = 1;
            while (remaining != 0) {
                std::vector<Real> next(n, 0);
                for (std::size_t c = 0; c < cells.size(); ++c) {
                    if (!supported_cell[c] || component.cell[c] != label) continue;
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
            unplaced_by_interface[label] += remaining;
            if (remaining == 0) ++redistributed;
            else ++skipped;
            return remaining;
        };
        for (std::size_t i = 0; i < n; ++i) {
            if (unplaced[i] != 0) {
                const auto label = component.node[i] >= 0 ? component.node[i] : -1 - static_cast<long long>(i);
                distribute_remaining(i, label, unplaced[i]);
            }
        }
        for (const auto& [origin, k] : component.splits) {
            const auto share = split_unplaced[k];
            if (share == 0) {
                continue;
            }
            const auto [source, label] = origin;
            if (used_connected_shared_patch) {
                if (distribute_remaining(source, label, share) != share) {
                    *used_connected_shared_patch = true;
                }
            } else {
                unplaced_by_interface[label] += share;
            }
        }
        for (const auto& [label, remaining] : unplaced_by_interface) unplaced_volume += std::abs(remaining);
        result.degeneracy_frozen_dof_indices.clear();
        for (std::size_t i = 0; i < n; ++i) {
            if (frozen_flag[i] > Real{0.0}) {
                ++frozen;
                result.degeneracy_frozen_dof_indices.push_back(i);
            }
        }
        Real change = 0.0;
        Real change_above_roundoff = 0.0;
        Real correction = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const Real value = transported_level_set[i] + delta[i];
            candidate[i] = value;
            const Real node_change = std::abs(value - iterate[i]);
            change = std::max(change, node_change);
            if (node_change > rate[i].transported_roundoff) {
                change_above_roundoff = std::max(change_above_roundoff, node_change);
            }
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
                same_class = cutClass(cells[c], trial, isovalue, geometry_tolerance, minimum_fraction) == transported_cut_class[c];
            }
            if (allTrue(context, same_class)) {
                candidate.swap(trial);
            }
        }
        prior_change = change;
        iterate.swap(candidate);
        result.iterations = iteration + 1;
        const bool nodal_convergence = change_above_roundoff <= kGeometryRelativeTolerance * correction;
        if (nodal_convergence || (stopped_above_nodal_roundoff && change <= field_roundoff.front())) {
            // The volume budget takes precedence: wall-trace nodes of a connected
            // interface that cannot otherwise meet it join the interior correction.
            const auto measured = computeLevelSetCutCellVolume(
                mesh, level_set_dofs, volume_options, std::span<const Real>(iterate.data(), iterate.size()));
            const Real limit = solver_tolerance(measured.negative_volume, correction_volume);
            bool released = false;
            for (std::size_t i = 0; i < n; ++i) {
                // Volume of a node outside every labelled interface is kept under
                // its own key -1 - i.
                const auto found = unplaced_by_interface.find(
                    component.node[i] >= 0 ? component.node[i] : -1 - static_cast<long long>(i));
                if (wall_node[i] && found != unplaced_by_interface.end() && !(std::abs(found->second) <= limit)) {
                    wall_node[i] = 0;
                    wall_delta[i] = 0;
                    ++result.wall_trace_released_dofs;
                    released = true;
                }
            }
            if (!released) {
                if (stopped_above_nodal_roundoff) {
                    *stopped_above_nodal_roundoff = !nodal_convergence && unplaced_volume <= limit;
                }
                result.converged = true;
                break;
            }
            prior_change = std::numeric_limits<Real>::infinity();
        }
    }
    InterfaceSums final_sums(n);
    if (!evaluate_endpoint(iterate, final_sums, true)) {
        return result;
    }
    result.current_interface_cells = static_cast<std::size_t>(current_cut_cells);
    result.sign_preserving_skipped_dofs = skipped;
    result.sign_preserving_redistributed_dofs = redistributed;
    result.degeneracy_frozen_dofs = frozen;

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

    const Real solver_volume_tolerance = solver_tolerance(reconciled_volume.negative_volume, correction_volume);
    result.volume_tolerance = std::abs(temporal_volume_remainder) + solver_volume_tolerance;
    if (!std::isfinite(result.reconciled_volume_error) || !std::isfinite(result.volume_tolerance) ||
        !std::isfinite(unplaced_volume)) {
        result.diagnostic = "level-set kinematic reconciliation produced a nonfinite volume budget";
        return result;
    }
    // Opposite infeasible fluxes on separate interfaces can sum to zero.
    // Their unsatisfied local volume constraints cannot cancel one another,
    // so every connected interface also meets its own budget.
    Real worst_component = 0;
    std::size_t worst_component_index = component.count;
    for (std::size_t c = 0; c < component.count; ++c) {
        const Real* terms = &budget[kBudgetTerms * c];
        const Real error = terms[0] - Real{0.5} * dt * (terms[1] + terms[2]);
        const Real allowed = std::abs(terms[3]) + kGeometryRelativeTolerance * correction_volume +
                             Real{128} * std::numeric_limits<Real>::epsilon() *
                                 (terms[4] + dt * (std::abs(terms[1]) + std::abs(terms[2])));
        if (!(std::abs(error) <= allowed) && std::abs(error) > worst_component) {
            worst_component = std::abs(error);
            worst_component_index = c;
        }
    }
    result.volume_balance_satisfied = std::abs(result.reconciled_volume_error) <= result.volume_tolerance &&
                                      unplaced_volume <= solver_volume_tolerance && worst_component == 0;
    if (!result.converged || !result.volume_balance_satisfied) {
        result.retryable_failure = true;
        std::ostringstream message;
        message.precision(std::numeric_limits<Real>::max_digits10);
        message << (result.converged ? "kinematic volume target unattainable on the constrained interface"
                                    : "kinematic fixed-point iteration limit")
                << ", sharp volume defect=" << result.reconciled_volume_error
                << ", tolerance=" << result.volume_tolerance
                << ", unplaced constrained volume=" << unplaced_volume
                << ", largest connected-interface defect=" << worst_component;
        if (worst_component_index < component.count) {
            // A translated closed interface can miss every vertex of its former
            // support. Distinguish loss of the retained P1 cut from a constraint
            // on an interface that still exists. Each rank sees only its cells.
            std::vector<Real> cut_present(2u, 0);
            for (std::size_t c = 0; c < cells.size(); ++c) {
                if (component.cell[c] != static_cast<long long>(worst_component_index)) {
                    continue;
                }
                if (cells[c].previous_has_surface) {
                    cut_present[0] = 1;
                }
                if (hasCutSurface(transported_cut_class[c])) {
                    cut_present[1] = 1;
                }
            }
            maxInPlace(context, cut_present);
            if (cut_present[0] > 0 && cut_present[1] == 0) {
                message << ", previously cut component has no retained P1 endpoint interface";
            }
        }
        result.diagnostic = message.str();
        return result;
    }
    // Return the caller's original coefficient on every frozen band vertex.
    if (transported_level_set.data() != original_transported.data()) {
        for (std::size_t i = 0; i < n; ++i) {
            if (transported_level_set[i] == isovalue) {
                iterate[i] = original_transported[i];
            }
        }
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
    const auto reconcile = [&](bool* used_connected_shared_patch) {
        const auto attempt = [&](bool* stopped_above_nodal_roundoff) {
            return integrateKinematicStep(mesh, level_set_dofs, velocity_dofs, isovalue, tolerance, dt,
                previous_level_set, transported_level_set, previous_velocity, transported_velocity,
                &reconciled_level_set, velocity_space, used_connected_shared_patch, stopped_above_nodal_roundoff);
        };
        bool stopped_above_nodal_roundoff = false;
        auto result = attempt(&stopped_above_nodal_roundoff);
        if (result.retryable_failure && stopped_above_nodal_roundoff) {
            // A remote large coefficient need not limit a cut node's precision.
            // Retry only when the measured candidate actually fails its balance.
            result = attempt(nullptr);
        }
        return result;
    };
    bool used_connected_shared_patch = false;
    auto result = reconcile(&used_connected_shared_patch);
    if (!result.success && used_connected_shared_patch) {
        // A distant trace can carry a linearized share yet approach its maximum
        // swept volume before the nonlinear budget closes. The one-ring solve
        // can then release its component's wall constraints. Restart it from
        // the transported field so the failed geometry cannot bias the solve.
        result = reconcile(nullptr);
    }
    return result;
}

std::vector<Real> levelSetRepresentationSlope(const assembly::IMeshAccess& mesh,
                                              const dofs::DofHandler& level_set_dofs,
                                              std::span<const Real> before,
                                              std::span<const Real> after)
{
    const auto context = collectiveFor(level_set_dofs);
    const auto n = static_cast<std::size_t>(level_set_dofs.getNumDofs());
    std::vector<Real> sums(2u * n, 0);
    std::string local_failure;
    try {
        const auto* map = level_set_dofs.getEntityDofMap();
        const int dimension = mesh.dimension();
        if (before.size() != n || after.size() != n || map == nullptr || (dimension != 2 && dimension != 3)) {
            throw std::invalid_argument("requires vertex-nodal level-set slices on a 2D or 3D mesh");
        }
        std::vector<GlobalIndex> nodes;
        std::vector<std::array<Real, 3>> coordinates;
        mesh.forEachOwnedCell([&](GlobalIndex cell_id) {
            SimplexCell cell;
            cell.corners = dimension + 1;
            mesh.getCellNodes(cell_id, nodes);
            mesh.getCellCoordinates(cell_id, coordinates);
            if (mesh.getCellType(cell_id) != (dimension == 2 ? ElementType::Triangle3 : ElementType::Tetra4) ||
                nodes.size() < static_cast<std::size_t>(cell.corners) ||
                coordinates.size() < static_cast<std::size_t>(cell.corners)) {
                throw std::invalid_argument("supports only Triangle3 (2D) and Tetra4 (3D) cells");
            }
            std::array<Real, 4> a{}, b{};
            bool changed = false;
            for (std::size_t k = 0; k < static_cast<std::size_t>(cell.corners); ++k) {
                cell.x[k] = coordinates[k];
                cell.dof[k] = static_cast<std::size_t>(map->getVertexDofs(nodes[k]).front());
                a[k] = before[cell.dof[k]];
                b[k] = after[cell.dof[k]];
                changed = changed || a[k] != b[k];
            }
            Point ga{}, gb{};
            if (!changed || !simplexGradient(cell, a, ga) || !simplexGradient(cell, b, gb) || !(norm(ga) > 0)) {
                return;
            }
            const auto e1 = sub(cell.x[1], cell.x[0]);
            const auto e2 = sub(cell.x[2], cell.x[0]);
            const Real measure = dimension == 2 ? norm(cross(e1, e2))
                                               : std::abs(dot(cross(e1, e2), sub(cell.x[3], cell.x[0])));
            for (std::size_t k = 0; k < static_cast<std::size_t>(cell.corners); ++k) {
                sums[cell.dof[k]] += measure * norm(gb) / norm(ga);
                sums[n + cell.dof[k]] += measure;
            }
        });
    } catch (const std::exception& error) {
        local_failure = error.what();
    }
    if (!allTrue(context, local_failure.empty())) {
        throw std::runtime_error(local_failure.empty()
                                     ? "another rank failed to measure the level-set representation slope"
                                     : "level-set representation slope " + local_failure);
    }
    sumInPlace(context, sums);
    std::vector<Real> slope(n, 1);
    for (std::size_t i = 0; i < n; ++i) {
        if (sums[n + i] > 0) slope[i] = sums[i] / sums[n + i];
    }
    return slope;
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
