#include "LevelSet/LevelSetInterfacePreservingRedistancing.h"

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
            "level-set zero-polygon redistancing exceeds the MPI count range");
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

using Point2 = std::array<Real, 2>;

struct CellData {
    std::array<std::size_t, 3> dof{};
    std::array<Point2, 3> x{};
};

// The owned affine Triangle3 cells with their DOFs and vertex coordinates.
[[nodiscard]] std::vector<CellData> ownedCells(const assembly::IMeshAccess& mesh,
                                               const dofs::DofHandler& dofs,
                                               std::size_t n)
{
    if (mesh.dimension() != 2) {
        throw std::invalid_argument("supports only 2D meshes (affine Triangle3 cells)");
    }
    const auto* map = dofs.getEntityDofMap();
    if (map == nullptr) {
        throw std::invalid_argument("requires a vertex-nodal level-set layout");
    }
    std::vector<CellData> cells;
    std::vector<GlobalIndex> nodes;
    mesh.forEachOwnedCell([&](GlobalIndex cell_id) {
        if (mesh.getCellType(cell_id) != ElementType::Triangle3 ||
            mesh.getCellGeometryOrder(cell_id) > 1) {
            throw std::invalid_argument("supports only affine Triangle3 cells");
        }
        mesh.getCellNodes(cell_id, nodes);
        if (nodes.size() < 3u) {
            throw std::invalid_argument("found an incomplete cell");
        }
        CellData cell;
        for (std::size_t k = 0; k < 3u; ++k) {
            const auto vdofs = map->getVertexDofs(nodes[k]);
            if (vdofs.size() != 1u || vdofs.front() < 0 ||
                static_cast<std::size_t>(vdofs.front()) >= n) {
                throw std::invalid_argument(
                    "requires exactly one scalar level-set DOF per vertex");
            }
            cell.dof[k] = static_cast<std::size_t>(vdofs.front());
            const auto c = mesh.getNodeCoordinates(nodes[k]);
            cell.x[k] = {static_cast<Real>(c[0]), static_cast<Real>(c[1])};
        }
        cells.push_back(cell);
    });
    return cells;
}

// Zero crossing of edge (a, b) in the orientation of increasing DOF number,
// so that both cells of an interior edge compute the same bits.
[[nodiscard]] Point2 crossing(const CellData& cell, std::size_t a, std::size_t b,
                              Real isovalue, std::span<const Real> values)
{
    if (cell.dof[b] < cell.dof[a]) {
        std::swap(a, b);
    }
    const Real fa = values[cell.dof[a]] - isovalue;
    const Real fb = values[cell.dof[b]] - isovalue;
    const Real s = fa / (fa - fb);
    // A root on a vertex is that vertex exactly, whichever edge reports it.
    if (fa == Real{0.0}) {
        return cell.x[a];
    }
    if (fb == Real{0.0}) {
        return cell.x[b];
    }
    return {cell.x[a][0] + s * (cell.x[b][0] - cell.x[a][0]),
            cell.x[a][1] + s * (cell.x[b][1] - cell.x[a][1])};
}

constexpr std::array<std::array<std::size_t, 2>, 3> kEdges{{{0u, 1u}, {1u, 2u}, {2u, 0u}}};

// Bit mask of the cut edges of a cell and their crossings.
[[nodiscard]] int cutEdges(const CellData& cell, Real isovalue,
                           std::span<const Real> values,
                           std::array<Point2, 3>& points)
{
    int mask = 0;
    for (std::size_t e = 0; e < 3u; ++e) {
        const auto a = kEdges[e][0];
        const auto b = kEdges[e][1];
        const bool na = values[cell.dof[a]] < isovalue;
        const bool nb = values[cell.dof[b]] < isovalue;
        if (na != nb) {
            mask |= 1 << e;
            points[e] = crossing(cell, a, b, isovalue, values);
        }
    }
    return mask;
}

// Segments (x0, y0, x1, y1) of the owned cut cells.
[[nodiscard]] std::vector<Real> localSegments(const std::vector<CellData>& cells,
                                              Real isovalue,
                                              std::span<const Real> values)
{
    std::vector<Real> out;
    std::array<Point2, 3> points{};
    for (const auto& cell : cells) {
        const int mask = cutEdges(cell, isovalue, values, points);
        if (mask == 0) {
            continue;
        }
        std::array<Point2, 2> ends{};
        std::size_t k = 0u;
        for (std::size_t e = 0; e < 3u; ++e) {
            if ((mask & (1 << e)) != 0 && k < 2u) {
                ends[k++] = points[e];
            }
        }
        if (k != 2u) {
            throw std::logic_error("a cut triangle must have two cut edges");
        }
        out.insert(out.end(), {ends[0][0], ends[0][1], ends[1][0], ends[1][1]});
    }
    return out;
}

struct Ray {
    Point2 origin{};
    Point2 direction{};
};

// End points that belong to a single segment, continued away from it.
[[nodiscard]] std::vector<Ray> openEnds(const std::vector<Real>& segments)
{
    struct End {
        Point2 p{};
        Point2 other{};
    };
    const std::size_t count = segments.size() / 4u;
    std::vector<End> ends;
    ends.reserve(2u * count);
    for (std::size_t s = 0; s < count; ++s) {
        const Point2 p0{segments[4u * s], segments[4u * s + 1u]};
        const Point2 p1{segments[4u * s + 2u], segments[4u * s + 3u]};
        ends.push_back({p0, p1});
        ends.push_back({p1, p0});
    }
    std::sort(ends.begin(), ends.end(), [](const End& a, const End& b) {
        if (a.p[0] != b.p[0]) {
            return a.p[0] < b.p[0];
        }
        if (a.p[1] != b.p[1]) {
            return a.p[1] < b.p[1];
        }
        if (a.other[0] != b.other[0]) {
            return a.other[0] < b.other[0];
        }
        return a.other[1] < b.other[1];
    });
    std::vector<Ray> rays;
    for (std::size_t i = 0; i < ends.size();) {
        std::size_t j = i + 1u;
        while (j < ends.size() && ends[j].p == ends[i].p) {
            ++j;
        }
        if (j - i == 1u) {
            const Real dx = ends[i].p[0] - ends[i].other[0];
            const Real dy = ends[i].p[1] - ends[i].other[1];
            const Real length = std::hypot(dx, dy);
            if (length > Real{0.0}) {
                rays.push_back({ends[i].p, {dx / length, dy / length}});
            } else {
                rays.push_back({ends[i].p, {Real{0.0}, Real{0.0}}});
            }
        }
        i = j;
    }
    return rays;
}

[[nodiscard]] Real distanceToSegments(const Point2& x,
                                      const std::vector<Real>& segments,
                                      const std::vector<Ray>& rays)
{
    Real best = std::numeric_limits<Real>::infinity();
    const std::size_t count = segments.size() / 4u;
    for (std::size_t s = 0; s < count; ++s) {
        const Real ax = segments[4u * s];
        const Real ay = segments[4u * s + 1u];
        const Real ux = segments[4u * s + 2u] - ax;
        const Real uy = segments[4u * s + 3u] - ay;
        const Real len2 = ux * ux + uy * uy;
        Real t = Real{0.0};
        if (len2 > Real{0.0}) {
            t = std::clamp(((x[0] - ax) * ux + (x[1] - ay) * uy) / len2,
                           Real{0.0}, Real{1.0});
        }
        best = std::min(best, std::hypot(x[0] - ax - t * ux, x[1] - ay - t * uy));
    }
    for (const auto& ray : rays) {
        const Real px = x[0] - ray.origin[0];
        const Real py = x[1] - ray.origin[1];
        const Real t = std::max(Real{0.0},
                                px * ray.direction[0] + py * ray.direction[1]);
        best = std::min(best, std::hypot(px - t * ray.direction[0],
                                         py - t * ray.direction[1]));
    }
    return best;
}

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

LevelSetZeroPolygonRedistancingResult redistanceLevelSetToZeroPolygon(
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
    std::vector<CellData> cells;
    std::vector<Real> segments;
    if (local_failure.empty()) {
        try {
            cells = ownedCells(mesh, level_set_dofs, n);
            segments = localSegments(cells, isovalue, level_set);
        } catch (const std::exception& error) {
            local_failure = error.what();
        }
    }
    if (!allTrue(context, local_failure.empty())) {
        result.diagnostic =
            local_failure.empty()
                ? "another rank failed level-set zero-polygon redistancing validation"
                : "level-set zero-polygon redistancing " + local_failure;
        return result;
    }
    segments = allGather(context, segments);
    redistanced_level_set.assign(level_set.begin(), level_set.end());
    result.interface_segments = segments.size() / 4u;
    if (result.interface_segments == 0u) {
        result.success = true;
        result.diagnostic = "no interface";
        return result;
    }
    const auto rays = openEnds(segments);
    result.open_ends = rays.size();

    // Every rank evaluates the vertices of its owned cells against the common
    // segment list; the minimum over ranks selects the (identical) value.
    std::vector<Real> distance(n, std::numeric_limits<Real>::infinity());
    for (const auto& cell : cells) {
        for (std::size_t k = 0; k < 3u; ++k) {
            const auto i = cell.dof[k];
            if (std::isfinite(distance[i])) {
                continue;
            }
            distance[i] = distanceToSegments(cell.x[k], segments, rays);
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

LevelSetZeroPolygonChange measureLevelSetZeroPolygonChange(
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
    std::vector<Real> first_segments;
    std::vector<Real> second_segments;
    if (local_failure.empty()) {
        try {
            const auto cells = ownedCells(mesh, level_set_dofs, n);
            std::array<Point2, 3> pa{};
            std::array<Point2, 3> pb{};
            for (const auto& cell : cells) {
                const int ma = cutEdges(cell, isovalue, first, pa);
                const int mb = cutEdges(cell, isovalue, second, pb);
                if (ma != mb) {
                    ++change.cells_with_changed_cut;
                }
                for (std::size_t e = 0; e < 3u; ++e) {
                    if ((ma & mb & (1 << e)) != 0) {
                        shift[0] = std::max(shift[0], std::hypot(pa[e][0] - pb[e][0],
                                                                 pa[e][1] - pb[e][1]));
                    }
                }
            }
            first_segments = localSegments(cells, isovalue, first);
            second_segments = localSegments(cells, isovalue, second);
        } catch (const std::exception& error) {
            local_failure = error.what();
        }
    }
    if (!allTrue(context, local_failure.empty())) {
        change.diagnostic =
            local_failure.empty()
                ? "another rank failed level-set zero-polygon change validation"
                : "level-set zero-polygon change " + local_failure;
        return change;
    }
    reduceReals(context, shift, /*minimum=*/false);
    sumCount(context, change.cells_with_changed_cut);
    change.max_crossing_shift = shift[0];
    const auto ends_a = openEnds(allGather(context, first_segments));
    const auto ends_b = openEnds(allGather(context, second_segments));
    change.open_ends_first = ends_a.size();
    change.open_ends_second = ends_b.size();
    for (const auto& a : ends_a) {
        Real nearest = std::numeric_limits<Real>::infinity();
        for (const auto& b : ends_b) {
            nearest = std::min(nearest, std::hypot(a.origin[0] - b.origin[0],
                                                   a.origin[1] - b.origin[1]));
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
