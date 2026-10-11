#include "LevelSet/LevelSetInterfacePreservingRedistancing3D.h"

#include "Dofs/EntityDofMap.h"

#if FE_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace svmp::FE::level_set {
namespace {

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

[[nodiscard]] int checkedCount(std::size_t size)
{
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "level-set zero-surface redistancing exceeds the MPI count range");
    }
    return static_cast<int>(size);
}
#endif

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

void reduceReals(const Collective& context, std::vector<Real>& values, bool minimum)
{
#if FE_HAS_MPI
    if (context.active && !values.empty()) {
        MPI_Allreduce(MPI_IN_PLACE, values.data(), checkedCount(values.size()),
                      realType(), minimum ? MPI_MIN : MPI_MAX,
                      context.communicator);
    }
#else
    (void)context;
    (void)values;
    (void)minimum;
#endif
}

void sumCount(const Collective& context, std::size_t& value)
{
#if FE_HAS_MPI
    if (context.active) {
        unsigned long long v = value;
        MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                      context.communicator);
        value = static_cast<std::size_t>(v);
    }
#else
    (void)context;
    (void)value;
#endif
}

// Concatenate the local arrays of every rank in rank order.
[[nodiscard]] std::vector<Real> allGather(const Collective& context,
                                          const std::vector<Real>& local)
{
#if FE_HAS_MPI
    if (context.active) {
        int size = 1;
        MPI_Comm_size(context.communicator, &size);
        int count = checkedCount(local.size());
        std::vector<int> counts(static_cast<std::size_t>(size), 0);
        MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                      context.communicator);
        std::vector<int> displacements(static_cast<std::size_t>(size), 0);
        std::size_t total = 0u;
        for (int r = 0; r < size; ++r) {
            displacements[static_cast<std::size_t>(r)] = checkedCount(total);
            total += static_cast<std::size_t>(counts[static_cast<std::size_t>(r)]);
        }
        std::vector<Real> global(total, Real{0.0});
        MPI_Allgatherv(local.data(), count, realType(), global.data(),
                       counts.data(), displacements.data(), realType(),
                       context.communicator);
        return global;
    }
#else
    (void)context;
#endif
    return local;
}

using Point3 = std::array<Real, 3>;

struct Tetra {
    std::array<std::size_t, 4> dof{};
    std::array<Point3, 4> x{};
    std::array<GlobalIndex, 4> vertex{};
};

// The owned affine Tetra4 cells with their DOFs and vertex coordinates.
[[nodiscard]] std::vector<Tetra> ownedTetra(const assembly::IMeshAccess& mesh,
                                            const dofs::DofHandler& dofs,
                                            std::size_t n)
{
    if (mesh.dimension() != 3) {
        throw std::invalid_argument("supports only 3D meshes (affine Tetra4 cells)");
    }
    const auto* map = dofs.getEntityDofMap();
    if (map == nullptr) {
        throw std::invalid_argument("requires a vertex-nodal level-set layout");
    }
    std::vector<Tetra> cells;
    std::vector<GlobalIndex> nodes;
    mesh.forEachOwnedCell([&](GlobalIndex cell_id) {
        if (mesh.getCellType(cell_id) != ElementType::Tetra4 ||
            mesh.getCellGeometryOrder(cell_id) > 1) {
            throw std::invalid_argument("supports only affine Tetra4 cells");
        }
        mesh.getCellNodes(cell_id, nodes);
        if (nodes.size() < 4u) {
            throw std::invalid_argument("found an incomplete cell");
        }
        Tetra cell;
        for (std::size_t k = 0; k < 4u; ++k) {
            const auto vdofs = map->getVertexDofs(nodes[k]);
            if (vdofs.size() != 1u || vdofs.front() < 0 ||
                static_cast<std::size_t>(vdofs.front()) >= n) {
                throw std::invalid_argument(
                    "requires exactly one scalar level-set DOF per vertex");
            }
            cell.dof[k] = static_cast<std::size_t>(vdofs.front());
            cell.vertex[k] = nodes[k];
            const auto c = mesh.getNodeCoordinates(nodes[k]);
            cell.x[k] = {static_cast<Real>(c[0]), static_cast<Real>(c[1]),
                         static_cast<Real>(c[2])};
        }
        cells.push_back(cell);
    });
    return cells;
}

constexpr std::array<std::array<std::size_t, 2>, 6> kEdges{
    {{0u, 1u}, {0u, 2u}, {0u, 3u}, {1u, 2u}, {1u, 3u}, {2u, 3u}}};

[[nodiscard]] std::size_t edgeIndex(std::size_t a, std::size_t b)
{
    for (std::size_t e = 0; e < 6u; ++e) {
        if ((kEdges[e][0] == a && kEdges[e][1] == b) ||
            (kEdges[e][0] == b && kEdges[e][1] == a)) {
            return e;
        }
    }
    throw std::logic_error("not a tetrahedron edge");
}

// Zero crossing of edge (a, b) in the orientation of increasing DOF number,
// so that every cell sharing the edge computes the same bits.
[[nodiscard]] Point3 crossing(const Tetra& cell, std::size_t a, std::size_t b,
                              Real isovalue, std::span<const Real> values)
{
    if (cell.dof[b] < cell.dof[a]) {
        std::swap(a, b);
    }
    const Real fa = values[cell.dof[a]] - isovalue;
    const Real fb = values[cell.dof[b]] - isovalue;
    // A root on a vertex is that vertex exactly, whichever edge reports it.
    if (fa == Real{0.0}) {
        return cell.x[a];
    }
    if (fb == Real{0.0}) {
        return cell.x[b];
    }
    const Real s = fa / (fa - fb);
    Point3 p{};
    for (std::size_t c = 0; c < 3u; ++c) {
        p[c] = cell.x[a][c] + s * (cell.x[b][c] - cell.x[a][c]);
    }
    return p;
}

// Bit mask of the cut edges of a cell and their crossings.
[[nodiscard]] int cutEdges(const Tetra& cell, Real isovalue,
                           std::span<const Real> values,
                           std::array<Point3, 6>& points)
{
    int mask = 0;
    for (std::size_t e = 0; e < 6u; ++e) {
        const auto a = kEdges[e][0];
        const auto b = kEdges[e][1];
        if ((values[cell.dof[a]] < isovalue) != (values[cell.dof[b]] < isovalue)) {
            mask |= 1 << e;
            points[e] = crossing(cell, a, b, isovalue, values);
        }
    }
    return mask;
}

void appendTriangle(std::vector<Real>& out, const Point3& p, const Point3& q, const Point3& r)
{
    // A triangle with two coincident corners (roots on vertices) has no area.
    if (p == q || q == r || r == p) {
        return;
    }
    out.insert(out.end(), {p[0], p[1], p[2], q[0], q[1], q[2], r[0], r[1], r[2]});
}

// Triangles (9 reals each) of the zero surface in the owned cut cells.
[[nodiscard]] std::vector<Real> localTriangles(const std::vector<Tetra>& cells,
                                               Real isovalue,
                                               std::span<const Real> values)
{
    std::vector<Real> out;
    std::array<Point3, 6> points{};
    for (const auto& cell : cells) {
        if (cutEdges(cell, isovalue, values, points) == 0) {
            continue;
        }
        std::array<bool, 4> below{};
        std::size_t count = 0u;
        for (std::size_t k = 0; k < 4u; ++k) {
            below[k] = values[cell.dof[k]] < isovalue;
            count += below[k] ? 1u : 0u;
        }
        if (count == 1u || count == 3u) {
            std::size_t lone = 0u;
            while (below[lone] != (count == 1u)) {
                ++lone;
            }
            std::array<Point3, 3> corner{};
            std::size_t m = 0u;
            for (std::size_t k = 0; k < 4u; ++k) {
                if (k != lone) {
                    corner[m++] = points[edgeIndex(lone, k)];
                }
            }
            appendTriangle(out, corner[0], corner[1], corner[2]);
            continue;
        }
        // Two vertices below (a, b), two above (c, d): the quadrilateral
        // p(ac), p(ad), p(bd), p(bc), split along the diagonal through the
        // crossing of the cut edge with the smallest DOF pair.
        std::array<std::size_t, 2> lo{};
        std::array<std::size_t, 2> hi{};
        std::size_t nl = 0u;
        std::size_t nh = 0u;
        for (std::size_t k = 0; k < 4u; ++k) {
            if (below[k]) {
                lo[nl++] = k;
            } else {
                hi[nh++] = k;
            }
        }
        const std::array<std::array<std::size_t, 2>, 4> quad{
            {{lo[0], hi[0]}, {lo[0], hi[1]}, {lo[1], hi[1]}, {lo[1], hi[0]}}};
        std::size_t first = 0u;
        auto key = [&](std::size_t q) {
            const auto da = cell.dof[quad[q][0]];
            const auto db = cell.dof[quad[q][1]];
            return std::make_pair(std::min(da, db), std::max(da, db));
        };
        for (std::size_t q = 1; q < 4u; ++q) {
            if (key(q) < key(first)) {
                first = q;
            }
        }
        std::array<Point3, 4> p{};
        for (std::size_t q = 0; q < 4u; ++q) {
            p[q] = points[edgeIndex(quad[(first + q) % 4u][0], quad[(first + q) % 4u][1])];
        }
        appendTriangle(out, p[0], p[1], p[2]);
        appendTriangle(out, p[0], p[2], p[3]);
    }
    return out;
}

[[nodiscard]] Point3 sub(const Point3& a, const Point3& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

[[nodiscard]] Real dot(const Point3& a, const Point3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] Point3 cross(const Point3& a, const Point3& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

[[nodiscard]] Real norm(const Point3& a)
{
    return std::sqrt(dot(a, a));
}

[[nodiscard]] Real pointSegmentDistance(const Point3& x, const Point3& a, const Point3& b)
{
    const auto u = sub(b, a);
    const auto w = sub(x, a);
    const Real len2 = dot(u, u);
    Real t = Real{0.0};
    if (len2 > Real{0.0}) {
        t = std::clamp(dot(w, u) / len2, Real{0.0}, Real{1.0});
    }
    return norm({w[0] - t * u[0], w[1] - t * u[1], w[2] - t * u[2]});
}

[[nodiscard]] Real pointTriangleDistance(const Point3& x, const Point3& a, const Point3& b,
                                         const Point3& c)
{
    const auto ab = sub(b, a);
    const auto ac = sub(c, a);
    const auto n = cross(ab, ac);
    const Real nn = dot(n, n);
    if (nn > Real{0.0}) {
        const auto ap = sub(x, a);
        const Real d00 = dot(ab, ab);
        const Real d01 = dot(ab, ac);
        const Real d11 = dot(ac, ac);
        const Real d20 = dot(ap, ab);
        const Real d21 = dot(ap, ac);
        const Real denominator = d00 * d11 - d01 * d01;
        if (denominator > Real{0.0}) {
            const Real v = (d11 * d20 - d01 * d21) / denominator;
            const Real w = (d00 * d21 - d01 * d20) / denominator;
            if (v >= Real{0.0} && w >= Real{0.0} && v + w <= Real{1.0}) {
                return std::abs(dot(ap, n)) / std::sqrt(nn);
            }
        }
    }
    return std::min({pointSegmentDistance(x, a, b), pointSegmentDistance(x, b, c),
                     pointSegmentDistance(x, c, a)});
}

// Continuation of a triangle beyond an open edge: the half-strip
// {origin + s*edge + t*outward : 0 <= s <= length, t >= 0} in its plane.
struct Strip {
    Point3 origin{};
    Point3 edge{};      // unit edge direction
    Point3 outward{};   // unit, in the triangle plane, perpendicular to edge, away from it
    Real length{0.0};
    bool has_strip{false};
};

struct OpenEdge {
    Point3 a{};
    Point3 b{};
    Point3 opposite{};
};

// Triangle edges used by a single triangle (compared bitwise).
[[nodiscard]] std::vector<OpenEdge> openEdges(const std::vector<Real>& triangles)
{
    struct Edge {
        Point3 lo{};
        Point3 hi{};
        Point3 opposite{};
    };
    const std::size_t count = triangles.size() / 9u;
    std::vector<Edge> edges;
    edges.reserve(3u * count);
    for (std::size_t t = 0; t < count; ++t) {
        std::array<Point3, 3> p{};
        for (std::size_t k = 0; k < 3u; ++k) {
            p[k] = {triangles[9u * t + 3u * k], triangles[9u * t + 3u * k + 1u],
                    triangles[9u * t + 3u * k + 2u]};
        }
        for (std::size_t k = 0; k < 3u; ++k) {
            const auto& u = p[k];
            const auto& v = p[(k + 1u) % 3u];
            edges.push_back({std::min(u, v), std::max(u, v), p[(k + 2u) % 3u]});
        }
    }
    std::sort(edges.begin(), edges.end(), [](const Edge& x, const Edge& y) {
        if (x.lo != y.lo) {
            return x.lo < y.lo;
        }
        if (x.hi != y.hi) {
            return x.hi < y.hi;
        }
        return x.opposite < y.opposite;
    });
    std::vector<OpenEdge> open;
    for (std::size_t i = 0; i < edges.size();) {
        std::size_t j = i + 1u;
        while (j < edges.size() && edges[j].lo == edges[i].lo && edges[j].hi == edges[i].hi) {
            ++j;
        }
        if (j - i == 1u) {
            open.push_back({edges[i].lo, edges[i].hi, edges[i].opposite});
        }
        i = j;
    }
    return open;
}

[[nodiscard]] std::vector<Strip> strips(const std::vector<OpenEdge>& open)
{
    std::vector<Strip> out;
    out.reserve(open.size());
    for (const auto& e : open) {
        Strip s;
        s.origin = e.a;
        const auto u = sub(e.b, e.a);
        s.length = norm(u);
        if (s.length > Real{0.0}) {
            s.edge = {u[0] / s.length, u[1] / s.length, u[2] / s.length};
            const auto w = sub(e.a, e.opposite);
            const Real along = dot(w, s.edge);
            const Point3 d{w[0] - along * s.edge[0], w[1] - along * s.edge[1],
                           w[2] - along * s.edge[2]};
            const Real dn = norm(d);
            if (dn > Real{0.0}) {
                s.outward = {d[0] / dn, d[1] / dn, d[2] / dn};
                s.has_strip = true;
            }
        }
        out.push_back(s);
    }
    return out;
}

[[nodiscard]] Real pointStripDistance(const Point3& x, const Strip& s)
{
    const auto w = sub(x, s.origin);
    const Real u = std::clamp(dot(w, s.edge), Real{0.0}, s.length);
    const Real t = std::max(Real{0.0}, dot(w, s.outward));
    return norm({w[0] - u * s.edge[0] - t * s.outward[0],
                 w[1] - u * s.edge[1] - t * s.outward[1],
                 w[2] - u * s.edge[2] - t * s.outward[2]});
}

// Exact minimum distance to the triangles and half-strips.  The bounding-box
// test only skips triangles that are farther than the current minimum by a
// margin far above round-off, so the result does not depend on the order of
// the triangle list.
class SurfaceDistance {
public:
    SurfaceDistance(const std::vector<Real>& triangles, std::vector<Strip> continuation)
        : triangles_(triangles), strips_(std::move(continuation))
    {
        const std::size_t count = triangles_.size() / 9u;
        boxes_.resize(6u * count);
        Real scale = Real{1.0};
        for (std::size_t t = 0; t < count; ++t) {
            for (std::size_t c = 0; c < 3u; ++c) {
                Real lo = triangles_[9u * t + c];
                Real hi = lo;
                for (std::size_t k = 1; k < 3u; ++k) {
                    lo = std::min(lo, triangles_[9u * t + 3u * k + c]);
                    hi = std::max(hi, triangles_[9u * t + 3u * k + c]);
                }
                boxes_[6u * t + 2u * c] = lo;
                boxes_[6u * t + 2u * c + 1u] = hi;
                scale = std::max({scale, std::abs(lo), std::abs(hi)});
            }
        }
        margin_ = Real{1.0e-10} * scale;
    }

    [[nodiscard]] Real operator()(const Point3& x) const
    {
        Real best = std::numeric_limits<Real>::infinity();
        for (const auto& s : strips_) {
            if (s.has_strip) {
                best = std::min(best, pointStripDistance(x, s));
            }
        }
        const std::size_t count = triangles_.size() / 9u;
        for (std::size_t t = 0; t < count; ++t) {
            Real gap2 = Real{0.0};
            for (std::size_t c = 0; c < 3u; ++c) {
                const Real lo = boxes_[6u * t + 2u * c];
                const Real hi = boxes_[6u * t + 2u * c + 1u];
                const Real g = x[c] < lo ? lo - x[c] : (x[c] > hi ? x[c] - hi : Real{0.0});
                gap2 += g * g;
            }
            if (std::sqrt(gap2) > best + margin_) {
                continue;
            }
            const Real* p = triangles_.data() + 9u * t;
            best = std::min(best, pointTriangleDistance(x, {p[0], p[1], p[2]}, {p[3], p[4], p[5]},
                                                        {p[6], p[7], p[8]}));
        }
        return best;
    }

private:
    const std::vector<Real>& triangles_;
    std::vector<Strip> strips_;
    std::vector<Real> boxes_;
    Real margin_{0.0};
};

[[nodiscard]] std::string validate(Real isovalue, std::span<const Real> values,
                                   std::size_t n)
{
    if (!std::isfinite(isovalue)) {
        return "requires a finite isovalue";
    }
    if (values.size() != n) {
        return "received a coefficient span that does not match the field layout";
    }
    for (const auto v : values) {
        if (!std::isfinite(v)) {
            return "received non-finite level-set coefficients";
        }
    }
    return {};
}

} // namespace

LevelSetZeroPolygonRedistancingResult redistanceLevelSetToZeroSurface(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    Real isovalue,
    std::span<const Real> level_set,
    std::vector<Real>& redistanced_level_set)
{
    LevelSetZeroPolygonRedistancingResult result;
    const auto context = collectiveFor(level_set_dofs);
    const auto n = static_cast<std::size_t>(level_set_dofs.getNumDofs());
    std::string local_failure = validate(isovalue, level_set, n);
    std::vector<Tetra> cells;
    std::vector<Real> triangles;
    if (local_failure.empty()) {
        try {
            cells = ownedTetra(mesh, level_set_dofs, n);
            triangles = localTriangles(cells, isovalue, level_set);
        } catch (const std::exception& error) {
            local_failure = error.what();
        }
    }
    if (!allTrue(context, local_failure.empty())) {
        result.diagnostic =
            local_failure.empty()
                ? "another rank failed level-set zero-surface redistancing validation"
                : "level-set zero-surface redistancing " + local_failure;
        return result;
    }
    triangles = allGather(context, triangles);
    redistanced_level_set.assign(level_set.begin(), level_set.end());
    result.interface_segments = triangles.size() / 9u;
    if (result.interface_segments == 0u) {
        result.success = true;
        result.diagnostic = "no interface";
        return result;
    }
    const auto open = openEdges(triangles);
    result.open_ends = open.size();
    const SurfaceDistance distance_to(triangles, strips(open));

    // Every rank evaluates the vertices of its owned cells against the common
    // triangle list; the minimum over ranks selects the (identical) value.
    std::vector<Real> distance(n, std::numeric_limits<Real>::infinity());
    for (const auto& cell : cells) {
        for (std::size_t k = 0; k < 4u; ++k) {
            const auto i = cell.dof[k];
            if (std::isfinite(distance[i])) {
                continue;
            }
            distance[i] = distance_to(cell.x[k]);
        }
    }
    reduceReals(context, distance, /*minimum=*/true);
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(distance[i])) {
            continue;
        }
        const Real value = level_set[i] < isovalue ? isovalue - distance[i]
                                                   : isovalue + distance[i];
        if (value != level_set[i]) {
            redistanced_level_set[i] = value;
            ++result.redistanced_dofs;
            result.max_abs_change =
                std::max(result.max_abs_change, std::abs(value - level_set[i]));
        }
    }
    result.applied = result.redistanced_dofs > 0u;
    result.success = true;
    result.diagnostic = result.applied ? "redistanced" : "already a distance";
    return result;
}

std::vector<std::size_t> levelSetDofCellClosure3D(const assembly::IMeshAccess& mesh,
                                                  const dofs::DofHandler& level_set_dofs,
                                                  std::span<const std::size_t> dofs)
{
    const auto context = collectiveFor(level_set_dofs);
    const auto n = static_cast<std::size_t>(level_set_dofs.getNumDofs());
    std::vector<Real> marked(n, Real{0.0});
    std::vector<char> seed(n, 0);
    for (const auto i : dofs) {
        if (i < n) {
            seed[i] = 1;
        }
    }
    std::string local_failure;
    try {
        for (const auto& cell : ownedTetra(mesh, level_set_dofs, n)) {
            if (seed[cell.dof[0]] || seed[cell.dof[1]] || seed[cell.dof[2]] ||
                seed[cell.dof[3]]) {
                for (const auto i : cell.dof) {
                    marked[i] = Real{1.0};
                }
            }
        }
    } catch (const std::exception& error) {
        local_failure = error.what();
    }
    if (!allTrue(context, local_failure.empty())) {
        throw std::runtime_error("level-set DOF cell closure " +
                                 (local_failure.empty() ? std::string("failed on another rank")
                                                        : local_failure));
    }
    reduceReals(context, marked, /*minimum=*/false);
    std::vector<std::size_t> closure;
    for (std::size_t i = 0; i < n; ++i) {
        if (marked[i] > Real{0.0} || seed[i]) {
            closure.push_back(i);
        }
    }
    return closure;
}

LevelSetInterfaceMotionScale levelSetInterfaceMotionScale3D(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const dofs::DofHandler& velocity_dofs,
    Real isovalue,
    std::span<const Real> level_set,
    std::span<const Real> velocity)
{
    LevelSetInterfaceMotionScale scale;
    const auto context = collectiveFor(level_set_dofs);
    const auto n = static_cast<std::size_t>(level_set_dofs.getNumDofs());
    std::string local_failure = validate(isovalue, level_set, n);
    std::vector<Real> extrema{Real{0.0}, -std::numeric_limits<Real>::infinity()};
    if (local_failure.empty()) {
        try {
            const auto* velocity_map = velocity_dofs.getEntityDofMap();
            if (velocity_map == nullptr ||
                velocity.size() != static_cast<std::size_t>(velocity_dofs.getNumDofs())) {
                throw std::invalid_argument("needs a vertex-nodal velocity layout");
            }
            for (const auto& cell : ownedTetra(mesh, level_set_dofs, n)) {
                bool below = false;
                bool above = false;
                for (const auto i : cell.dof) {
                    below = below || level_set[i] < isovalue;
                    above = above || level_set[i] >= isovalue;
                }
                if (!(below && above)) {
                    continue;
                }
                for (std::size_t k = 0; k < 4u; ++k) {
                    const auto vdofs = velocity_map->getVertexDofs(cell.vertex[k]);
                    Real speed2 = 0.0;
                    for (std::size_t c = 0; c < vdofs.size() && c < 3u; ++c) {
                        const auto d = static_cast<std::size_t>(vdofs[c]);
                        if (d >= velocity.size()) {
                            throw std::invalid_argument("velocity DOF outside the coefficient span");
                        }
                        speed2 += velocity[d] * velocity[d];
                    }
                    extrema[0] = std::max(extrema[0], std::sqrt(speed2));
                }
                for (const auto& e : kEdges) {
                    // Stored as a negative value so that one MAX reduction
                    // yields both the largest speed and the smallest edge.
                    extrema[1] = std::max(extrema[1], -norm(sub(cell.x[e[0]], cell.x[e[1]])));
                }
            }
        } catch (const std::exception& error) {
            local_failure = error.what();
        }
    }
    if (!allTrue(context, local_failure.empty())) {
        scale.diagnostic = local_failure.empty()
                               ? "another rank failed level-set motion-scale validation"
                               : "level-set interface motion scale " + local_failure;
        return scale;
    }
    reduceReals(context, extrema, /*minimum=*/false);
    scale.max_speed = extrema[0];
    scale.min_edge = std::isfinite(extrema[1]) ? -extrema[1] : Real{0.0};
    scale.success = true;
    scale.diagnostic = "measured";
    return scale;
}

LevelSetZeroPolygonChange measureLevelSetZeroSurfaceChange(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    Real isovalue,
    std::span<const Real> first,
    std::span<const Real> second)
{
    LevelSetZeroPolygonChange change;
    const auto context = collectiveFor(level_set_dofs);
    const auto n = static_cast<std::size_t>(level_set_dofs.getNumDofs());
    std::string local_failure = validate(isovalue, first, n);
    if (local_failure.empty()) {
        local_failure = validate(isovalue, second, n);
    }
    std::vector<Real> shift{Real{0.0}};
    std::vector<Real> first_triangles;
    std::vector<Real> second_triangles;
    if (local_failure.empty()) {
        try {
            const auto cells = ownedTetra(mesh, level_set_dofs, n);
            std::array<Point3, 6> pa{};
            std::array<Point3, 6> pb{};
            for (const auto& cell : cells) {
                const int ma = cutEdges(cell, isovalue, first, pa);
                const int mb = cutEdges(cell, isovalue, second, pb);
                if (ma != mb) {
                    ++change.cells_with_changed_cut;
                }
                for (std::size_t e = 0; e < 6u; ++e) {
                    if ((ma & mb & (1 << e)) != 0) {
                        shift[0] = std::max(shift[0], norm(sub(pa[e], pb[e])));
                    }
                }
            }
            first_triangles = localTriangles(cells, isovalue, first);
            second_triangles = localTriangles(cells, isovalue, second);
        } catch (const std::exception& error) {
            local_failure = error.what();
        }
    }
    if (!allTrue(context, local_failure.empty())) {
        change.diagnostic =
            local_failure.empty()
                ? "another rank failed level-set zero-surface change validation"
                : "level-set zero-surface change " + local_failure;
        return change;
    }
    reduceReals(context, shift, /*minimum=*/false);
    sumCount(context, change.cells_with_changed_cut);
    change.max_crossing_shift = shift[0];
    const auto open_a = openEdges(allGather(context, first_triangles));
    const auto open_b = openEdges(allGather(context, second_triangles));
    change.open_ends_first = open_a.size();
    change.open_ends_second = open_b.size();
    for (const auto& a : open_a) {
        const Point3 mid{Real{0.5} * (a.a[0] + a.b[0]), Real{0.5} * (a.a[1] + a.b[1]),
                         Real{0.5} * (a.a[2] + a.b[2])};
        Real nearest = std::numeric_limits<Real>::infinity();
        for (const auto& b : open_b) {
            nearest = std::min(nearest, pointSegmentDistance(mid, b.a, b.b));
        }
        if (std::isfinite(nearest)) {
            change.max_open_end_shift = std::max(change.max_open_end_shift, nearest);
        }
    }
    change.success = true;
    change.diagnostic = "measured";
    return change;
}

} // namespace svmp::FE::level_set
