#include "LevelSet/LevelSetVolume.h"

#include "Dofs/EntityDofMap.h"
#include "Geometry/MappingFactory.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/FunctionSpace.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp::FE::level_set {
namespace {

using interfaces::CutInterfaceDomainRequest;
using interfaces::LevelSetCellCutInput;
using interfaces::LevelSetInterfaceSource;

struct CollectiveContext {
#if FE_HAS_MPI
    MPI_Comm communicator{MPI_COMM_NULL};
#endif
    bool active{false};
};

[[nodiscard]] CollectiveContext collectiveContext(
    const dofs::DofHandler& dof_handler)
{
    CollectiveContext context;
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
[[nodiscard]] MPI_Datatype mpiRealType() noexcept
{
    if constexpr (std::is_same_v<Real, float>) {
        return MPI_FLOAT;
    }
    return MPI_DOUBLE;
}
#endif

[[nodiscard]] Real allReduceRealMin(const CollectiveContext& context,
                                    Real local)
{
#if FE_HAS_MPI
    if (context.active) {
        Real global = local;
        MPI_Allreduce(&local,
                      &global,
                      1,
                      mpiRealType(),
                      MPI_MIN,
                      context.communicator);
        return global;
    }
#else
    (void)context;
#endif
    return local;
}

[[nodiscard]] Real allReduceRealMax(const CollectiveContext& context,
                                    Real local)
{
#if FE_HAS_MPI
    if (context.active) {
        Real global = local;
        MPI_Allreduce(&local,
                      &global,
                      1,
                      mpiRealType(),
                      MPI_MAX,
                      context.communicator);
        return global;
    }
#else
    (void)context;
#endif
    return local;
}

[[nodiscard]] bool allReduceLogicalAnd(const CollectiveContext& context,
                                       bool local)
{
#if FE_HAS_MPI
    if (context.active) {
        const int local_value = local ? 1 : 0;
        int global_value = 0;
        MPI_Allreduce(&local_value,
                      &global_value,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      context.communicator);
        return global_value != 0;
    }
#else
    (void)context;
#endif
    return local;
}

void globalizeVolumeResult(LevelSetVolumeResult& result,
                           const CollectiveContext& context)
{
    result.success = allReduceLogicalAnd(context, result.success);
#if FE_HAS_MPI
    if (context.active) {
        const std::array<unsigned long long, 4> local_counts{{
            static_cast<unsigned long long>(result.cells),
            static_cast<unsigned long long>(result.cut_cells),
            static_cast<unsigned long long>(result.full_negative_cells),
            static_cast<unsigned long long>(result.full_positive_cells),
        }};
        std::array<unsigned long long, 4> global_counts{};
        MPI_Allreduce(local_counts.data(),
                      global_counts.data(),
                      static_cast<int>(local_counts.size()),
                      MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM,
                      context.communicator);
        result.cells = static_cast<std::size_t>(global_counts[0]);
        result.cut_cells = static_cast<std::size_t>(global_counts[1]);
        result.full_negative_cells =
            static_cast<std::size_t>(global_counts[2]);
        result.full_positive_cells =
            static_cast<std::size_t>(global_counts[3]);

        const std::array<Real, 3> local_volumes{{
            result.total_volume,
            result.negative_volume,
            result.positive_volume,
        }};
        std::array<Real, 3> global_volumes{};
        MPI_Allreduce(local_volumes.data(),
                      global_volumes.data(),
                      static_cast<int>(local_volumes.size()),
                      mpiRealType(),
                      MPI_SUM,
                      context.communicator);
        result.total_volume = global_volumes[0];
        result.negative_volume = global_volumes[1];
        result.positive_volume = global_volumes[2];
    }
#else
    (void)context;
#endif
}

[[nodiscard]] std::array<Real, 3> sub(const std::array<Real, 3>& a,
                                      const std::array<Real, 3>& b) noexcept
{
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

[[nodiscard]] std::array<Real, 3> add(const std::array<Real, 3>& a,
                                      const std::array<Real, 3>& b) noexcept
{
    return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
}

[[nodiscard]] std::array<Real, 3> scale(const std::array<Real, 3>& a,
                                        Real s) noexcept
{
    return {{a[0] * s, a[1] * s, a[2] * s}};
}

[[nodiscard]] std::array<Real, 3> cross(const std::array<Real, 3>& a,
                                        const std::array<Real, 3>& b) noexcept
{
    return {{a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]}};
}

[[nodiscard]] Real dot(const std::array<Real, 3>& a,
                       const std::array<Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] Real norm(const std::array<Real, 3>& a) noexcept
{
    return std::sqrt(dot(a, a));
}

[[nodiscard]] Real triangleArea(const std::array<Real, 3>& a,
                                const std::array<Real, 3>& b,
                                const std::array<Real, 3>& c) noexcept
{
    return Real{0.5} * norm(cross(sub(b, a), sub(c, a)));
}

[[nodiscard]] Real tetraVolume(const std::array<Real, 3>& a,
                               const std::array<Real, 3>& b,
                               const std::array<Real, 3>& c,
                               const std::array<Real, 3>& d) noexcept
{
    return std::abs(dot(sub(b, a), cross(sub(c, a), sub(d, a)))) / Real{6.0};
}

[[nodiscard]] std::array<Real, 3> interpolatePoint(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b,
    Real t) noexcept
{
    return add(scale(a, Real{1.0} - t), scale(b, t));
}

[[nodiscard]] Real clampVolume(Real value, Real measure) noexcept
{
    if (value <= Real{0.0}) {
        return Real{0.0};
    }
    if (value >= measure) {
        return measure;
    }
    return value;
}

[[nodiscard]] std::size_t cornerCount(ElementType type)
{
    switch (type) {
    case ElementType::Triangle3:
    case ElementType::Triangle6:
        return 3u;
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return 4u;
    case ElementType::Hex8:
    case ElementType::Hex20:
    case ElementType::Hex27:
        return 8u;
    case ElementType::Wedge6:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
        return 6u;
    case ElementType::Pyramid5:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14:
        return 5u;
    default:
        return 0u;
    }
}

[[nodiscard]] std::vector<std::array<std::size_t, 2>> cornerEdgePairs(
    ElementType type)
{
    switch (type) {
    case ElementType::Triangle3:
    case ElementType::Triangle6:
        return {{{0u, 1u}}, {{1u, 2u}}, {{2u, 0u}}};
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
        return {{{0u, 1u}}, {{1u, 2u}}, {{2u, 3u}}, {{3u, 0u}}};
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return {{{0u, 1u}}, {{0u, 2u}}, {{0u, 3u}},
                {{1u, 2u}}, {{1u, 3u}}, {{2u, 3u}}};
    case ElementType::Hex8:
    case ElementType::Hex20:
    case ElementType::Hex27:
        return {{{0u, 1u}}, {{1u, 2u}}, {{2u, 3u}}, {{3u, 0u}},
                {{4u, 5u}}, {{5u, 6u}}, {{6u, 7u}}, {{7u, 4u}},
                {{0u, 4u}}, {{1u, 5u}}, {{2u, 6u}}, {{3u, 7u}}};
    case ElementType::Wedge6:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
        return {{{0u, 1u}}, {{1u, 2u}}, {{2u, 0u}},
                {{3u, 4u}}, {{4u, 5u}}, {{5u, 3u}},
                {{0u, 3u}}, {{1u, 4u}}, {{2u, 5u}}};
    case ElementType::Pyramid5:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14:
        return {{{0u, 1u}}, {{1u, 2u}}, {{2u, 3u}}, {{3u, 0u}},
                {{0u, 4u}}, {{1u, 4u}}, {{2u, 4u}}, {{3u, 4u}}};
    default:
        return {};
    }
}

struct ShiftDisplacementMetrics {
    Real minimum_edge_length{std::numeric_limits<Real>::infinity()};
    Real max_interface_displacement{0.0};
    Real max_contact_line_displacement{0.0};
    std::size_t crossing_count{0u};
    std::size_t contact_line_fragment_count{0u};
};

[[nodiscard]] std::vector<std::size_t> localFaceCornerIndices(
    ElementType type,
    LocalIndex face)
{
    const int f = static_cast<int>(face);
    switch (type) {
    case ElementType::Triangle3:
    case ElementType::Triangle6:
        if (f == 0) return {0u, 1u};
        if (f == 1) return {1u, 2u};
        if (f == 2) return {2u, 0u};
        return {};
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
        if (f == 0) return {0u, 1u};
        if (f == 1) return {1u, 2u};
        if (f == 2) return {2u, 3u};
        if (f == 3) return {3u, 0u};
        return {};
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        if (f == 0) return {0u, 2u, 1u};
        if (f == 1) return {0u, 1u, 3u};
        if (f == 2) return {1u, 2u, 3u};
        if (f == 3) return {2u, 0u, 3u};
        return {};
    case ElementType::Hex8:
    case ElementType::Hex20:
    case ElementType::Hex27:
        if (f == 0) return {0u, 3u, 2u, 1u};
        if (f == 1) return {4u, 5u, 6u, 7u};
        if (f == 2) return {0u, 1u, 5u, 4u};
        if (f == 3) return {1u, 2u, 6u, 5u};
        if (f == 4) return {2u, 3u, 7u, 6u};
        if (f == 5) return {3u, 0u, 4u, 7u};
        return {};
    case ElementType::Wedge6:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
        if (f == 0) return {0u, 1u, 2u};
        if (f == 1) return {3u, 4u, 5u};
        if (f == 2) return {0u, 1u, 4u, 3u};
        if (f == 3) return {1u, 2u, 5u, 4u};
        if (f == 4) return {2u, 0u, 3u, 5u};
        return {};
    case ElementType::Pyramid5:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14:
        if (f == 0) return {0u, 1u, 2u, 3u};
        if (f == 1) return {0u, 1u, 4u};
        if (f == 2) return {1u, 2u, 4u};
        if (f == 3) return {2u, 3u, 4u};
        if (f == 4) return {3u, 0u, 4u};
        return {};
    default:
        return {};
    }
}

[[nodiscard]] Real tangentialLinearGradientNorm(
    const std::array<Real, 3>& x0,
    const std::array<Real, 3>& x1,
    const std::array<Real, 3>& x2,
    Real value0,
    Real value1,
    Real value2) noexcept
{
    const auto d1 = sub(x1, x0);
    const auto d2 = sub(x2, x0);
    const Real g11 = dot(d1, d1);
    const Real g12 = dot(d1, d2);
    const Real g22 = dot(d2, d2);
    const Real determinant = g11 * g22 - g12 * g12;
    if (!(determinant > Real{0.0}) || !std::isfinite(determinant)) {
        return Real{0.0};
    }
    const Real dv1 = value1 - value0;
    const Real dv2 = value2 - value0;
    const Real alpha = (dv1 * g22 - dv2 * g12) / determinant;
    const Real beta = (dv2 * g11 - dv1 * g12) / determinant;
    return norm(add(scale(d1, alpha), scale(d2, beta)));
}

[[nodiscard]] ShiftDisplacementMetrics shiftDisplacementMetrics(
    const assembly::IMeshAccess& mesh,
    const dofs::EntityDofMap& entity_map,
    const LevelSetVolumeOptions& options,
    std::span<const Real> coefficients,
    Real shift,
    const CollectiveContext& collective)
{
    ShiftDisplacementMetrics metrics;
    std::vector<GlobalIndex> nodes;
    std::vector<std::array<Real, 3>> points;
    mesh.forEachOwnedCell([&](GlobalIndex cell) {
        const auto count = cornerCount(mesh.getCellType(cell));
        mesh.getCellNodes(cell, nodes);
        mesh.getCellCoordinates(cell, points);
        if (nodes.size() < count || points.size() < count) {
            return;
        }
        for (const auto edge : cornerEdgePairs(mesh.getCellType(cell))) {
            if (edge[0] >= count || edge[1] >= count) {
                continue;
            }
            const Real edge_length = norm(sub(points[edge[1]], points[edge[0]]));
            if (std::isfinite(edge_length) && edge_length > Real{0.0}) {
                metrics.minimum_edge_length =
                    std::min(metrics.minimum_edge_length, edge_length);
            }
            const auto da = entity_map.getVertexDofs(nodes[edge[0]]);
            const auto db = entity_map.getVertexDofs(nodes[edge[1]]);
            if (da.size() != 1u || db.size() != 1u ||
                da.front() < 0 || db.front() < 0 ||
                static_cast<std::size_t>(da.front()) >= coefficients.size() ||
                static_cast<std::size_t>(db.front()) >= coefficients.size()) {
                continue;
            }
            const Real a = coefficients[static_cast<std::size_t>(da.front())] -
                           options.isovalue;
            const Real b = coefficients[static_cast<std::size_t>(db.front())] -
                           options.isovalue;
            if (!((a < -options.tolerance && b > options.tolerance) ||
                  (a > options.tolerance && b < -options.tolerance))) {
                continue;
            }
            const Real denom = a - b;
            if (std::abs(denom) <= options.tolerance) {
                continue;
            }
            const Real old_t = a / denom;
            const Real new_t = (a + shift) / denom;
            const Real displacement = std::abs(new_t - old_t) * edge_length;
            metrics.max_interface_displacement =
                std::max(metrics.max_interface_displacement, displacement);
            ++metrics.crossing_count;
        }
    });

    // A wall contact line is a level-set zero contour restricted to a
    // boundary face.  In the low-level utility wall classifications are not
    // available, so use the maximum over every physical-boundary intersection
    // as a conservative wall-contact-line bound.  Volume-edge crossings alone
    // are not conservative when the interface meets a wall at a shallow angle.
    mesh.forEachBoundaryFace(
        -1,
        [&](GlobalIndex face, GlobalIndex cell) {
            if (!mesh.isOwnedCell(cell)) {
                return;
            }
            const auto type = mesh.getCellType(cell);
            const auto local_face = mesh.getLocalFaceIndex(face, cell);
            const auto face_corners = localFaceCornerIndices(type, local_face);
            if (face_corners.size() < 2u) {
                return;
            }
            std::vector<GlobalIndex> face_cell_nodes;
            std::vector<std::array<Real, 3>> face_cell_points;
            mesh.getCellNodes(cell, face_cell_nodes);
            mesh.getCellCoordinates(cell, face_cell_points);
            std::vector<Real> values;
            values.reserve(face_corners.size());
            for (const auto corner : face_corners) {
                if (corner >= face_cell_nodes.size() ||
                    corner >= face_cell_points.size()) {
                    return;
                }
                const auto dofs = entity_map.getVertexDofs(
                    face_cell_nodes[corner]);
                if (dofs.size() != 1u || dofs.front() < 0 ||
                    static_cast<std::size_t>(dofs.front()) >=
                        coefficients.size()) {
                    return;
                }
                values.push_back(
                    coefficients[static_cast<std::size_t>(dofs.front())] -
                    options.isovalue);
            }
            if (mesh.dimension() == 2) {
                const Real a = values[0];
                const Real b = values[1];
                if (!((a < -options.tolerance && b > options.tolerance) ||
                      (a > options.tolerance && b < -options.tolerance))) {
                    return;
                }
                const Real denominator = std::abs(a - b);
                if (!(denominator > options.tolerance)) {
                    return;
                }
                const Real edge_length = norm(sub(
                    face_cell_points[face_corners[1]],
                    face_cell_points[face_corners[0]]));
                metrics.max_contact_line_displacement = std::max(
                    metrics.max_contact_line_displacement,
                    std::abs(shift) * edge_length / denominator);
                ++metrics.contact_line_fragment_count;
                return;
            }
            if (mesh.dimension() != 3) {
                return;
            }
            for (std::size_t i = 1u; i + 1u < face_corners.size(); ++i) {
                const std::array<std::size_t, 3> triangle{{0u, i, i + 1u}};
                Real minimum = values[triangle[0]];
                Real maximum = values[triangle[0]];
                for (std::size_t j = 1u; j < triangle.size(); ++j) {
                    minimum = std::min(minimum, values[triangle[j]]);
                    maximum = std::max(maximum, values[triangle[j]]);
                }
                if (!(minimum < -options.tolerance &&
                      maximum > options.tolerance)) {
                    continue;
                }
                const Real gradient_norm = tangentialLinearGradientNorm(
                    face_cell_points[face_corners[triangle[0]]],
                    face_cell_points[face_corners[triangle[1]]],
                    face_cell_points[face_corners[triangle[2]]],
                    values[triangle[0]],
                    values[triangle[1]],
                    values[triangle[2]]);
                if (!(gradient_norm > options.tolerance)) {
                    continue;
                }
                metrics.max_contact_line_displacement = std::max(
                    metrics.max_contact_line_displacement,
                    std::abs(shift) / gradient_norm);
                ++metrics.contact_line_fragment_count;
            }
        });
    metrics.minimum_edge_length =
        allReduceRealMin(collective, metrics.minimum_edge_length);
#if FE_HAS_MPI
    if (collective.active) {
        const std::array<Real, 2> local_maxima{{
            metrics.max_interface_displacement,
            metrics.max_contact_line_displacement,
        }};
        std::array<Real, 2> global_maxima{};
        MPI_Allreduce(local_maxima.data(),
                      global_maxima.data(),
                      static_cast<int>(local_maxima.size()),
                      mpiRealType(),
                      MPI_MAX,
                      collective.communicator);
        metrics.max_interface_displacement = global_maxima[0];
        metrics.max_contact_line_displacement = global_maxima[1];
    }
#endif
    if (!std::isfinite(metrics.minimum_edge_length)) {
        metrics.minimum_edge_length = Real{0.0};
    }
    return metrics;
}

[[nodiscard]] Real limitShiftByInterfaceDisplacement(
    const assembly::IMeshAccess& mesh,
    const dofs::EntityDofMap& entity_map,
    const LevelSetVolumeOptions& volume_options,
    std::span<const Real> coefficients,
    Real requested_shift,
    Real maximum_displacement,
    const CollectiveContext& collective)
{
    if (!(maximum_displacement > Real{0.0}) ||
        requested_shift == Real{0.0}) {
        return requested_shift;
    }
    const auto requested = shiftDisplacementMetrics(
        mesh,
        entity_map,
        volume_options,
        coefficients,
        requested_shift,
        collective);
    const auto requested_max = std::max(
        requested.max_interface_displacement,
        requested.max_contact_line_displacement);
    if (requested_max <= maximum_displacement) {
        return requested_shift;
    }
    // Every displacement estimate above is homogeneous in |shift| for the
    // fixed original zero-set crossings.  Scale the requested shift directly;
    // this is the exact bound and avoids dozens of collective bisection probes.
    return requested_shift * (maximum_displacement / requested_max);
}

[[nodiscard]] std::vector<std::array<std::size_t, 4>>
tetrahedralCornerDecomposition(ElementType type)
{
    switch (type) {
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return {{{0u, 1u, 2u, 3u}}};
    case ElementType::Hex8:
    case ElementType::Hex20:
    case ElementType::Hex27:
        return {{{0u, 1u, 2u, 6u}},
                {{0u, 2u, 3u, 6u}},
                {{0u, 3u, 7u, 6u}},
                {{0u, 7u, 4u, 6u}},
                {{0u, 4u, 5u, 6u}},
                {{0u, 5u, 1u, 6u}}};
    case ElementType::Wedge6:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
        return {{{0u, 1u, 2u, 5u}},
                {{0u, 1u, 5u, 4u}},
                {{0u, 4u, 5u, 3u}}};
    case ElementType::Pyramid5:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14:
        return {{{0u, 1u, 2u, 4u}},
                {{0u, 2u, 3u, 4u}}};
    default:
        return {};
    }
}

[[nodiscard]] Real parentMeasure(ElementType type,
                                 const std::vector<std::array<Real, 3>>& x)
{
    switch (type) {
    case ElementType::Triangle3:
    case ElementType::Triangle6:
        return triangleArea(x[0], x[1], x[2]);
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
        return triangleArea(x[0], x[1], x[2]) + triangleArea(x[0], x[2], x[3]);
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return tetraVolume(x[0], x[1], x[2], x[3]);
    case ElementType::Hex8:
    case ElementType::Hex20:
    case ElementType::Hex27:
    case ElementType::Wedge6:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
    case ElementType::Pyramid5:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14: {
        Real volume = Real{0.0};
        for (const auto& tet : tetrahedralCornerDecomposition(type)) {
            volume += tetraVolume(x[tet[0]], x[tet[1]], x[tet[2]], x[tet[3]]);
        }
        return volume;
    }
    default:
        return Real{0.0};
    }
}

[[nodiscard]] Real coefficientAtVertex(const dofs::EntityDofMap& entity_map,
                                       GlobalIndex vertex,
                                       std::span<const Real> coefficients)
{
    const auto dofs = entity_map.getVertexDofs(vertex);
    if (dofs.size() != 1u) {
        throw std::invalid_argument(
            "level-set volume calculation requires one scalar DOF per mesh vertex");
    }
    const auto dof = dofs.front();
    if (dof < 0 || static_cast<std::size_t>(dof) >= coefficients.size()) {
        throw std::invalid_argument(
            "level-set volume calculation found a vertex DOF outside the coefficient span");
    }
    return coefficients[static_cast<std::size_t>(dof)];
}

[[nodiscard]] Real negativeFractionFromValues(std::span<const Real> values,
                                              Real tolerance)
{
    std::size_t negative_count = 0u;
    std::size_t positive_count = 0u;
    for (const auto value : values) {
        if (value < -tolerance) {
            ++negative_count;
        } else if (value > tolerance) {
            ++positive_count;
        }
    }
    if (positive_count == 0u) {
        return Real{1.0};
    }
    if (negative_count == 0u) {
        return Real{0.0};
    }
    return Real{-1.0};
}

[[nodiscard]] std::pair<Real, Real> vertexCoefficientRange(
    const assembly::IMeshAccess& mesh,
    const dofs::EntityDofMap& entity_map,
    std::span<const Real> coefficients,
    const CollectiveContext& collective)
{
    Real min_value = std::numeric_limits<Real>::infinity();
    Real max_value = -std::numeric_limits<Real>::infinity();
    std::set<GlobalIndex> owned_cell_vertices;
    std::vector<GlobalIndex> cell_nodes;
    mesh.forEachOwnedCell([&](GlobalIndex cell) {
        mesh.getCellNodes(cell, cell_nodes);
        owned_cell_vertices.insert(cell_nodes.begin(), cell_nodes.end());
    });
    for (const auto vertex : owned_cell_vertices) {
        const Real value = coefficientAtVertex(entity_map, vertex, coefficients);
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }
    min_value = allReduceRealMin(collective, min_value);
    max_value = allReduceRealMax(collective, max_value);
    if (!std::isfinite(min_value) || !std::isfinite(max_value)) {
        throw std::invalid_argument("level-set volume correction requires finite coefficients");
    }
    return {min_value, max_value};
}

[[nodiscard]] std::vector<Real> shiftedCoefficients(std::span<const Real> coefficients,
                                                    Real shift)
{
    std::vector<Real> shifted(coefficients.begin(), coefficients.end());
    for (auto& value : shifted) {
        value += shift;
    }
    return shifted;
}

[[nodiscard]] std::pair<Real, Real> coefficientRange(
    std::span<const Real> coefficients,
    const CollectiveContext& collective)
{
    Real min_value = std::numeric_limits<Real>::infinity();
    Real max_value = -std::numeric_limits<Real>::infinity();
    bool finite = true;
    for (const auto value : coefficients) {
        finite = finite && std::isfinite(value);
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }
    finite = allReduceLogicalAnd(collective, finite);
    min_value = allReduceRealMin(collective, min_value);
    max_value = allReduceRealMax(collective, max_value);
    if (!finite || !std::isfinite(min_value) || !std::isfinite(max_value)) {
        throw std::invalid_argument("level-set volume correction requires finite coefficients");
    }
    return {min_value, max_value};
}

[[nodiscard]] std::vector<Real> shiftedSystemSolution(
    std::span<const Real> solution,
    std::size_t offset,
    std::size_t field_dof_count,
    Real shift)
{
    std::vector<Real> shifted(solution.begin(), solution.end());
    for (std::size_t i = 0; i < field_dof_count; ++i) {
        shifted[offset + i] += shift;
    }
    return shifted;
}

[[nodiscard]] bool hasScalarContinuousSimplicialLinearLayout(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const CollectiveContext& collective)
{
    const auto* entity_map = level_set_dofs.getEntityDofMap();
    bool qualified = entity_map != nullptr;
    if (entity_map != nullptr) {
        const auto stats = entity_map->getStatistics();
        qualified = qualified && entity_map->numVertices() == mesh.numVertices() &&
                    stats.n_vertex_dofs == entity_map->numVertices() &&
                    stats.n_edge_dofs == 0 && stats.n_face_dofs == 0 &&
                    stats.n_cell_interior_dofs == 0;

        for (GlobalIndex vertex = 0;
             qualified && vertex < mesh.numVertices();
             ++vertex) {
            qualified = entity_map->getVertexDofs(vertex).size() == 1u;
        }

        std::vector<GlobalIndex> nodes;
        mesh.forEachCell([&](GlobalIndex cell) {
            if (!qualified) {
                return;
            }
            const auto cell_type = mesh.getCellType(cell);
            const bool supported_simplex =
                cell_type == ElementType::Triangle3 ||
                cell_type == ElementType::Tetra4;
            const auto corner_count = cornerCount(cell_type);
            mesh.getCellNodes(cell, nodes);
            const auto cell_dofs = level_set_dofs.getCellDofs(cell);
            if (!supported_simplex || mesh.getCellGeometryOrder(cell) != 1 ||
                corner_count == 0u || nodes.size() < corner_count ||
                cell_dofs.size() != corner_count) {
                qualified = false;
                return;
            }

            std::vector<GlobalIndex> expected;
            expected.reserve(corner_count);
            for (std::size_t corner = 0; corner < corner_count; ++corner) {
                const auto vertex_dofs = entity_map->getVertexDofs(nodes[corner]);
                if (vertex_dofs.size() != 1u) {
                    qualified = false;
                    return;
                }
                expected.push_back(vertex_dofs.front());
            }
            std::sort(expected.begin(), expected.end());
            std::vector<GlobalIndex> actual(cell_dofs.begin(), cell_dofs.end());
            std::sort(actual.begin(), actual.end());
            qualified = actual == expected;
        });
    }
    return allReduceLogicalAnd(collective, qualified);
}

void requireQualifiedGlobalShiftLayout(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const CollectiveContext& collective)
{
    if (!hasScalarContinuousSimplicialLinearLayout(
            mesh, level_set_dofs, collective)) {
        throw std::invalid_argument(
            "level-set global shift correction is qualified only for scalar "
            "continuous vertex-nodal P1 fields on linear triangles or "
            "tetrahedra; tensor-product, curved-geometry, high-order, "
            "hierarchical, discontinuous, vector, and non-nodal layouts are "
            "rejected before correction");
    }
}

void requireQualifiedGlobalShiftField(
    const systems::FESystem& system,
    FieldId level_set_field)
{
    const auto& field = system.fieldRecord(level_set_field);
    const auto& field_dofs = system.fieldDofHandler(level_set_field);
    const auto collective = collectiveContext(field_dofs);
    const auto* space = field.space.get();
    const bool qualified_space =
        space != nullptr && field.components == 1 &&
        space->space_type() == spaces::SpaceType::H1 &&
        space->field_type() == FieldType::Scalar &&
        space->continuity() == Continuity::C0 &&
        space->value_dimension() == 1 && !space->is_variable_order() &&
        space->polynomial_order() == 1 &&
        space->element().basis().basis_type() == BasisType::Lagrange;
    if (!allReduceLogicalAnd(collective, qualified_space)) {
        throw std::invalid_argument(
            "level-set global shift correction is qualified only for scalar "
            "continuous nodal first-order H1 Lagrange fields (simplicial P1)");
    }
    requireQualifiedGlobalShiftLayout(
        system.meshAccess(), field_dofs, collective);
}

struct TopologyStableShiftMetrics {
    bool valid{true};
    bool has_strict_crossing{false};
    Real maximum_symmetric_shift{std::numeric_limits<Real>::infinity()};
};

[[nodiscard]] TopologyStableShiftMetrics topologyStableShiftMetrics(
    const assembly::IMeshAccess& mesh,
    const dofs::EntityDofMap& entity_map,
    const LevelSetVolumeOptions& options,
    std::span<const Real> coefficients,
    const CollectiveContext& collective)
{
    TopologyStableShiftMetrics metrics;
    std::vector<GlobalIndex> nodes;
    mesh.forEachOwnedCell([&](GlobalIndex cell) {
        const auto count = cornerCount(mesh.getCellType(cell));
        mesh.getCellNodes(cell, nodes);
        if (count == 0u || nodes.size() < count) {
            metrics.valid = false;
            return;
        }

        std::vector<Real> values;
        values.reserve(count);
        for (std::size_t corner = 0; corner < count; ++corner) {
            const auto vertex_dofs = entity_map.getVertexDofs(nodes[corner]);
            if (vertex_dofs.size() != 1u || vertex_dofs.front() < 0 ||
                static_cast<std::size_t>(vertex_dofs.front()) >=
                    coefficients.size()) {
                metrics.valid = false;
                return;
            }
            const Real value =
                coefficients[static_cast<std::size_t>(vertex_dofs.front())] -
                options.isovalue;
            if (!std::isfinite(value) ||
                !(std::abs(value) > Real{2.0} * options.tolerance)) {
                metrics.valid = false;
                return;
            }
            metrics.maximum_symmetric_shift = std::min(
                metrics.maximum_symmetric_shift,
                std::abs(value) - Real{2.0} * options.tolerance);
            values.push_back(value);
        }

        for (const auto edge : cornerEdgePairs(mesh.getCellType(cell))) {
            if (edge[0] < values.size() && edge[1] < values.size() &&
                ((values[edge[0]] < -options.tolerance &&
                  values[edge[1]] > options.tolerance) ||
                 (values[edge[0]] > options.tolerance &&
                  values[edge[1]] < -options.tolerance))) {
                metrics.has_strict_crossing = true;
            }
        }
    });

    metrics.valid = allReduceLogicalAnd(collective, metrics.valid);
    metrics.maximum_symmetric_shift = allReduceRealMin(
        collective, metrics.maximum_symmetric_shift);
    metrics.has_strict_crossing =
        allReduceRealMax(
            collective, metrics.has_strict_crossing ? Real{1.0} : Real{0.0}) >
        Real{0.5};
    if (!metrics.valid || !metrics.has_strict_crossing ||
        !(metrics.maximum_symmetric_shift > Real{0.0}) ||
        !std::isfinite(metrics.maximum_symmetric_shift)) {
        metrics.valid = false;
        metrics.maximum_symmetric_shift = Real{0.0};
        return metrics;
    }
    metrics.maximum_symmetric_shift = std::nextafter(
        metrics.maximum_symmetric_shift, Real{0.0});
    metrics.valid = metrics.maximum_symmetric_shift > Real{0.0};
    return metrics;
}

[[nodiscard]] Real requireTopologyStableShiftBound(
    const assembly::IMeshAccess& mesh,
    const dofs::EntityDofMap& entity_map,
    const LevelSetVolumeOptions& options,
    std::span<const Real> coefficients,
    const CollectiveContext& collective)
{
    const auto metrics = topologyStableShiftMetrics(
        mesh, entity_map, options, coefficients, collective);
    if (!metrics.valid) {
        throw std::invalid_argument(
            "level-set global shift correction requires a nondegenerate "
            "simplicial P1 interface with at least one strict edge crossing "
            "and every vertex outside a two-tolerance safety margin around "
            "the isovalue");
    }
    return metrics.maximum_symmetric_shift;
}

[[nodiscard]] Real negativeVolumeForLinearTetra(
    const CutInterfaceDomainRequest& request,
    GlobalIndex parent_cell,
    std::span<const std::array<Real, 3>, 4> coordinates,
    std::span<const Real, 4> signed_values,
    Real tolerance)
{
    const Real measure = tetraVolume(
        coordinates[0], coordinates[1], coordinates[2], coordinates[3]);
    const Real simple_fraction =
        negativeFractionFromValues(signed_values, tolerance);
    if (simple_fraction >= Real{0.0}) {
        return simple_fraction * measure;
    }

    auto clipped_volume = [&]() {
        std::array<std::size_t, 4> inside{};
        std::array<std::size_t, 4> outside{};
        std::size_t inside_count = 0u;
        std::size_t outside_count = 0u;
        for (std::size_t i = 0; i < signed_values.size(); ++i) {
            if (signed_values[i] <= tolerance) {
                inside[inside_count++] = i;
            } else {
                outside[outside_count++] = i;
            }
        }
        if (inside_count == 0u) {
            return Real{0.0};
        }
        if (outside_count == 0u) {
            return measure;
        }

        auto interface_point = [&](std::size_t inside_index,
                                   std::size_t outside_index) {
            const Real vi = signed_values[inside_index];
            const Real vj = signed_values[outside_index];
            if (std::abs(vi) <= tolerance) {
                return coordinates[inside_index];
            }
            if (std::abs(vj) <= tolerance) {
                return coordinates[outside_index];
            }
            const Real denom = vi - vj;
            Real t = denom != Real{0.0} ? vi / denom : Real{0.0};
            t = std::clamp(t, Real{0.0}, Real{1.0});
            return interpolatePoint(coordinates[inside_index],
                                    coordinates[outside_index],
                                    t);
        };

        if (inside_count == 1u) {
            const auto a = inside[0];
            const auto p0 = interface_point(a, outside[0]);
            const auto p1 = interface_point(a, outside[1]);
            const auto p2 = interface_point(a, outside[2]);
            return clampVolume(
                tetraVolume(coordinates[a], p0, p1, p2),
                measure);
        }
        if (inside_count == 3u) {
            const auto p = outside[0];
            const auto q0 = interface_point(inside[0], p);
            const auto q1 = interface_point(inside[1], p);
            const auto q2 = interface_point(inside[2], p);
            return clampVolume(
                measure - tetraVolume(coordinates[p], q0, q1, q2),
                measure);
        }

        const auto a = inside[0];
        const auto b = inside[1];
        const auto c = outside[0];
        const auto d = outside[1];
        const auto p_ac = interface_point(a, c);
        const auto p_ad = interface_point(a, d);
        const auto p_bc = interface_point(b, c);
        const auto p_bd = interface_point(b, d);
        const Real volume =
            tetraVolume(coordinates[a], coordinates[b], p_bd, p_bc) +
            tetraVolume(coordinates[a], p_ac, p_bc, p_bd) +
            tetraVolume(coordinates[a], p_ac, p_bd, p_ad);
        return clampVolume(volume, measure);
    };

    LevelSetCellCutInput input{};
    input.parent_cell = parent_cell;
    input.element_type = ElementType::Tetra4;
    input.node_coordinates.assign(coordinates.begin(), coordinates.end());
    input.level_set_values.reserve(signed_values.size());
    for (const auto value : signed_values) {
        input.level_set_values.push_back(value + request.isovalue);
    }

    auto cut_result = interfaces::cutLinearLevelSetCell3D(request, input);
    auto active = std::find_if(
        cut_result.fragments.begin(),
        cut_result.fragments.end(),
        [](const auto& fragment) { return fragment.active(); });
    if (active == cut_result.fragments.end()) {
        return clipped_volume();
    }
    return active->negative_volume_fraction * measure;
}

[[nodiscard]] LevelSetGeneratedInterfaceOptions generatedInterfaceOptionsForVolume(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetVolumeOptions& options)
{
    LevelSetGeneratedInterfaceOptions generated;
    generated.level_set_field_name =
        options.level_set_field_name.empty()
            ? system.fieldRecord(level_set_field).name
            : options.level_set_field_name;
    generated.domain_id = options.generated_domain_id.empty()
                              ? std::string{"volume_correction"}
                              : options.generated_domain_id;
    generated.requested_interface_marker = options.requested_interface_marker;
    generated.isovalue = options.isovalue;
    generated.tolerance = options.tolerance;
    if (options.quadrature_order.has_value()) {
        generated.quadrature_order = *options.quadrature_order;
    }
    if (options.interface_quadrature_order.has_value()) {
        generated.interface_quadrature_order =
            *options.interface_quadrature_order;
    }
    if (options.volume_quadrature_order.has_value()) {
        generated.volume_quadrature_order = *options.volume_quadrature_order;
    }
    generated.geometry_mode = options.geometry_mode;
    generated.implicit_cut_quadrature_backend =
        options.implicit_cut_quadrature_backend;
    generated.implicit_cut_fallback_policy =
        options.implicit_cut_fallback_policy;
    generated.geometry_tangent_policy = options.geometry_tangent_policy;
    generated.implicit_cut_root_tolerance =
        options.implicit_cut_root_tolerance;
    generated.implicit_cut_root_coordinate_tolerance =
        options.implicit_cut_root_coordinate_tolerance;
    generated.implicit_cut_root_max_iterations =
        options.implicit_cut_root_max_iterations;
    generated.implicit_cut_max_subdivision_depth =
        options.implicit_cut_max_subdivision_depth;
    generated.affected_cell_neighborhood_layers =
        options.affected_cell_neighborhood_layers;
    generated.allow_corner_linearized_geometry =
        options.allow_corner_linearized_geometry;
    generated.require_production_qualified_implicit_cut_backend =
        options.require_production_qualified_implicit_cut_backend;
    return generated;
}

void populateGeneratedVolumeDiagnostics(
    LevelSetVolumeResult& result,
    const LevelSetGeneratedInterfaceResult& generated)
{
    result.generated_value_revision = generated.value_revision;
    result.generated_cell_cache_hits = generated.cell_cache_hits;
    result.generated_cell_cache_misses = generated.cell_cache_misses;
    result.generated_cell_cache_unchanged_dof_hits =
        generated.cell_cache_unchanged_dof_hits;
    result.generated_cell_refresh_candidate_count =
        generated.cell_refresh_candidate_count;
    result.generated_directly_affected_cell_count =
        generated.directly_affected_cell_count;
    result.generated_affected_cell_neighborhood_count =
        generated.affected_cell_neighborhood_count;
    result.generated_domain_cache_hits = generated.domain_cache_hits;
    result.generated_linear_full_cell_fast_path_count =
        generated.linear_full_cell_fast_path_count;
    result.generated_backend_elapsed_seconds =
        generated.backend_elapsed_seconds;
}

void accumulateGeneratedVolumeDiagnostics(
    LevelSetGlobalShiftCorrectionResult& correction,
    const LevelSetVolumeResult& volume)
{
    ++correction.generated_volume_measurement_count;
    correction.generated_cell_cache_hits += volume.generated_cell_cache_hits;
    correction.generated_cell_cache_misses += volume.generated_cell_cache_misses;
    correction.generated_cell_cache_unchanged_dof_hits +=
        volume.generated_cell_cache_unchanged_dof_hits;
    correction.generated_cell_refresh_candidate_count +=
        volume.generated_cell_refresh_candidate_count;
    correction.generated_directly_affected_cell_count +=
        volume.generated_directly_affected_cell_count;
    correction.generated_affected_cell_neighborhood_count +=
        volume.generated_affected_cell_neighborhood_count;
    correction.generated_domain_cache_hits += volume.generated_domain_cache_hits;
    correction.generated_linear_full_cell_fast_path_count +=
        volume.generated_linear_full_cell_fast_path_count;
    correction.generated_backend_elapsed_seconds +=
        volume.generated_backend_elapsed_seconds;
}

[[nodiscard]] std::shared_ptr<geometry::GeometryMapping>
makeGeneratedVolumeCellMapping(const assembly::IMeshAccess& mesh,
                               GlobalIndex cell)
{
    std::vector<std::array<Real, 3>> coordinates;
    mesh.getCellCoordinates(cell, coordinates);
    if (coordinates.empty()) {
        throw std::runtime_error(
            "generated level-set volume found a cell with no geometry coordinates");
    }
    std::vector<math::Vector<Real, 3>> nodes;
    nodes.reserve(coordinates.size());
    for (const auto& coordinate : coordinates) {
        nodes.push_back(math::Vector<Real, 3>{
            coordinate[0], coordinate[1], coordinate[2]});
    }
    geometry::MappingRequest request;
    request.element_type = mesh.getCellType(cell);
    request.geometry_order = mesh.getCellGeometryOrder(cell);
    request.use_affine = request.geometry_order <= 1;
    return geometry::MappingFactory::create(request, nodes);
}

[[nodiscard]] bool isFullGeneratedVolumeRule(
    const geometry::CutQuadratureRule& rule) noexcept
{
    if (rule.kind != geometry::CutQuadratureKind::Volume ||
        !rule.full_cell_equivalent || !std::isfinite(rule.measure) ||
        !std::isfinite(rule.parent_measure) ||
        !std::isfinite(rule.volume_fraction) ||
        !(rule.parent_measure > Real{0.0})) {
        return false;
    }
    const Real tolerance = Real{128.0} *
                           std::numeric_limits<Real>::epsilon();
    return std::abs(rule.volume_fraction - Real{1.0}) <= tolerance &&
           std::abs(rule.measure - rule.parent_measure) <=
               tolerance * std::max(Real{1.0}, std::abs(rule.parent_measure));
}

[[nodiscard]] Real physicalGeneratedVolumeRuleMeasure(
    const assembly::IMeshAccess& mesh,
    const geometry::CutQuadratureRule& rule,
    geometry::GeometryMapping& mapping)
{
    if (rule.kind != geometry::CutQuadratureKind::Volume) {
        throw std::invalid_argument(
            "generated level-set volume requires a volume quadrature rule");
    }
    if (rule.frame == geometry::CutGeometryFrame::Current) {
        if (!std::isfinite(rule.measure) || rule.measure < Real{0.0}) {
            throw std::runtime_error(
                "generated level-set volume received an invalid current-frame measure");
        }
        return rule.measure;
    }

    Real measure{0.0};
    if (isFullGeneratedVolumeRule(rule)) {
        const auto cell = static_cast<GlobalIndex>(
            rule.provenance.parent_entity);
        const int geometry_order =
            std::max(1, mesh.getCellGeometryOrder(cell));
        const auto full_rule = quadrature::QuadratureFactory::create(
            mesh.getCellType(cell), std::max(2, 2 * geometry_order));
        for (std::size_t q = 0; q < full_rule->num_points(); ++q) {
            const auto xi = full_rule->point(q);
            const Real determinant = mapping.jacobian_determinant(xi);
            const Real weight = full_rule->weight(q);
            if (!std::isfinite(determinant) || !std::isfinite(weight)) {
                throw std::runtime_error(
                    "generated level-set volume encountered a non-finite full-cell mapping");
            }
            measure += weight * std::abs(determinant);
        }
        return measure;
    }

    if (rule.points.empty()) {
        throw std::runtime_error(
            "generated level-set volume received an empty partial rule");
    }
    for (const auto& point : rule.points) {
        const math::Vector<Real, 3> xi{
            point.point[0], point.point[1], point.point[2]};
        const Real determinant = mapping.jacobian_determinant(xi);
        if (!std::isfinite(determinant) || !std::isfinite(point.weight) ||
            point.weight < Real{0.0}) {
            throw std::runtime_error(
                "generated level-set volume encountered an invalid pointwise mapping");
        }
        measure += point.weight * std::abs(determinant);
    }
    return measure;
}

[[nodiscard]] LevelSetVolumeResult computeGeneratedInterfaceLevelSetVolume(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetVolumeOptions& options,
    std::span<const Real> solution,
    LevelSetGeneratedInterfaceLifecycle& lifecycle)
{
    LevelSetVolumeResult result;
    const auto& field_dofs = system.fieldDofHandler(level_set_field);
    const auto collective = collectiveContext(field_dofs);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(level_set_field));
    if (offset + n_field_dofs > solution.size()) {
        throw std::invalid_argument(
            "level-set volume calculation received an incompatible system solution span");
    }

    const auto generated_options =
        generatedInterfaceOptionsForVolume(system, level_set_field, options);
    const auto generated =
        lifecycle.build(system, generated_options, solution);
    populateGeneratedVolumeDiagnostics(result, generated);
    const bool generated_success =
        allReduceLogicalAnd(collective, generated.success);
    if (!generated_success) {
        result.success = false;
        result.diagnostic = generated.success
                                ? "generated-interface volume construction failed on another MPI rank"
                                : generated.diagnostic;
        globalizeVolumeResult(result, collective);
        return result;
    }

    result.success = true;
    system.meshAccess().forEachOwnedCell(
        [&](GlobalIndex /*cell*/) { ++result.cells; });
    result.diagnostic = "generated_interface_quadrature";

    std::set<GlobalIndex> negative_cells;
    std::set<GlobalIndex> positive_cells;
    std::set<GlobalIndex> cut_cells;
    std::unordered_map<GlobalIndex,
                       std::shared_ptr<geometry::GeometryMapping>>
        mapping_by_cell;
    for (const auto& fragment : generated.domain.fragments()) {
        if (fragment.active() &&
            system.meshAccess().isOwnedCell(fragment.parent_cell)) {
            cut_cells.insert(fragment.parent_cell);
        }
    }
    for (const auto& rule : generated.domain.volumeQuadratureRules()) {
        if (rule.kind != geometry::CutQuadratureKind::Volume) {
            continue;
        }
        const auto parent_cell = static_cast<GlobalIndex>(
            rule.provenance.parent_entity);
        if (!system.meshAccess().isOwnedCell(parent_cell)) {
            continue;
        }
        auto [mapping, inserted] = mapping_by_cell.try_emplace(
            parent_cell, nullptr);
        if (inserted) {
            mapping->second = makeGeneratedVolumeCellMapping(
                system.meshAccess(), parent_cell);
        }
        const Real physical_measure = physicalGeneratedVolumeRuleMeasure(
            system.meshAccess(), rule, *mapping->second);
        result.total_volume += physical_measure;
        if (rule.side == geometry::CutIntegrationSide::Negative) {
            negative_cells.insert(parent_cell);
            result.negative_volume += physical_measure;
        } else if (rule.side == geometry::CutIntegrationSide::Positive) {
            positive_cells.insert(parent_cell);
            result.positive_volume += physical_measure;
        }
    }
    for (const auto cell : negative_cells) {
        if (positive_cells.find(cell) != positive_cells.end()) {
            cut_cells.insert(cell);
        }
    }
    result.cut_cells = cut_cells.size();
    for (const auto cell : negative_cells) {
        if (cut_cells.find(cell) == cut_cells.end()) {
            ++result.full_negative_cells;
        }
    }
    for (const auto cell : positive_cells) {
        if (cut_cells.find(cell) == cut_cells.end()) {
            ++result.full_positive_cells;
        }
    }
    globalizeVolumeResult(result, collective);
    return result;
}

[[nodiscard]] LevelSetVolumeResult computeGeneratedInterfaceLevelSetVolume(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetVolumeOptions& options,
    std::span<const Real> solution)
{
    LevelSetGeneratedInterfaceLifecycle lifecycle;
    return computeGeneratedInterfaceLevelSetVolume(
        system,
        level_set_field,
        options,
        solution,
        lifecycle);
}

} // namespace

LevelSetVolumeResult computeLevelSetCutCellVolume(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetVolumeOptions& options,
    std::span<const Real> coefficients)
{
    if (options.use_generated_interface_quadrature) {
        throw std::invalid_argument(
            "generated-interface level-set volume calculation requires the FESystem overload");
    }
    if (!(options.tolerance > 0.0) || !std::isfinite(options.tolerance) ||
        !std::isfinite(options.isovalue)) {
        throw std::invalid_argument(
            "level-set volume calculation requires a finite isovalue and positive finite tolerance");
    }
    const auto expected = static_cast<std::size_t>(level_set_dofs.getNumDofs());
    if (coefficients.size() != expected) {
        throw std::invalid_argument(
            "level-set volume calculation received an incompatible coefficient span");
    }
    const auto* entity_map = level_set_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::invalid_argument("level-set volume calculation requires a scalar nodal field");
    }
    if (entity_map->numVertices() != mesh.numVertices()) {
        throw std::invalid_argument(
            "level-set volume calculation requires field and mesh vertex counts to match");
    }
    const auto collective = collectiveContext(level_set_dofs);

    CutInterfaceDomainRequest request{};
    request.source = LevelSetInterfaceSource::fromField(FieldId{0});
    request.interface_marker = 0;
    request.isovalue = options.isovalue;
    request.tolerance = options.tolerance;
    request.quadrature_order = 1;

    LevelSetVolumeResult result;
    result.success = true;

    std::vector<GlobalIndex> cell_nodes;
    std::vector<std::array<Real, 3>> cell_coordinates;
    mesh.forEachOwnedCell([&](GlobalIndex cell_id) {
        const auto type = mesh.getCellType(cell_id);
        const std::size_t count = cornerCount(type);
        if (count == 0u) {
            throw std::invalid_argument(
                "level-set volume calculation encountered an unsupported element type");
        }

        mesh.getCellNodes(cell_id, cell_nodes);
        mesh.getCellCoordinates(cell_id, cell_coordinates);
        if (cell_nodes.size() < count || cell_coordinates.size() < count) {
            throw std::invalid_argument(
                "level-set volume calculation found incomplete cell geometry");
        }

        std::vector<Real> signed_values;
        signed_values.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            signed_values.push_back(
                coefficientAtVertex(*entity_map, cell_nodes[i], coefficients) -
                options.isovalue);
        }

        const auto measure = parentMeasure(type, cell_coordinates);
        result.total_volume += measure;
        ++result.cells;

        const Real simple_fraction =
            negativeFractionFromValues(signed_values, options.tolerance);
        Real negative_fraction = simple_fraction;
        if (simple_fraction < Real{0.0}) {
            if (mesh.dimension() == 2) {
                LevelSetCellCutInput input{};
                input.parent_cell = cell_id;
                input.element_type = type;
                input.node_coordinates.assign(
                    cell_coordinates.begin(),
                    cell_coordinates.begin() + static_cast<std::ptrdiff_t>(count));
                input.level_set_values.reserve(count);
                for (const auto value : signed_values) {
                    input.level_set_values.push_back(value + options.isovalue);
                }
                auto cut_result =
                    interfaces::cutLinearLevelSetCell2D(request, input);
                auto active = std::find_if(
                    cut_result.fragments.begin(),
                    cut_result.fragments.end(),
                    [](const auto& fragment) { return fragment.active(); });
                if (active == cut_result.fragments.end()) {
                    throw std::runtime_error(
                        "level-set volume calculation found a cut cell without an active interface fragment");
                }
                negative_fraction = active->negative_volume_fraction;
                ++result.cut_cells;
            } else if (mesh.dimension() == 3) {
                const auto tets = tetrahedralCornerDecomposition(type);
                if (tets.empty()) {
                    throw std::invalid_argument(
                        "level-set volume calculation encountered an unsupported 3D element type");
                }
                Real negative_volume = Real{0.0};
                for (const auto& tet : tets) {
                    const std::array<std::array<Real, 3>, 4> tet_coordinates{{
                        cell_coordinates[tet[0]],
                        cell_coordinates[tet[1]],
                        cell_coordinates[tet[2]],
                        cell_coordinates[tet[3]],
                    }};
                    const std::array<Real, 4> tet_values{{
                        signed_values[tet[0]],
                        signed_values[tet[1]],
                        signed_values[tet[2]],
                        signed_values[tet[3]],
                    }};
                    negative_volume += negativeVolumeForLinearTetra(
                        request,
                        cell_id,
                        std::span<const std::array<Real, 3>, 4>(
                            tet_coordinates.data(), tet_coordinates.size()),
                        std::span<const Real, 4>(
                            tet_values.data(), tet_values.size()),
                        options.tolerance);
                }
                negative_fraction =
                    measure > Real{0.0} ? negative_volume / measure : Real{0.0};
                ++result.cut_cells;
            } else {
                throw std::invalid_argument(
                    "level-set volume calculation requires a 2D or 3D mesh");
            }
        } else if (negative_fraction >= Real{1.0}) {
            ++result.full_negative_cells;
        } else {
            ++result.full_positive_cells;
        }

        const Real negative = negative_fraction * measure;
        result.negative_volume += negative;
        result.positive_volume += measure - negative;
    });

    globalizeVolumeResult(result, collective);
    return result;
}

LevelSetVolumeResult computeLevelSetCutCellVolume(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetVolumeOptions& options,
    std::span<const Real> solution)
{
    if (options.use_generated_interface_quadrature) {
        return computeGeneratedInterfaceLevelSetVolume(
            system,
            level_set_field,
            options,
            solution);
    }

    const auto& field_dofs = system.fieldDofHandler(level_set_field);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(level_set_field));
    if (offset + n_field_dofs > solution.size()) {
        throw std::invalid_argument(
            "level-set volume calculation received an incompatible system solution span");
    }

    return computeLevelSetCutCellVolume(
        system.meshAccess(),
        field_dofs,
        options,
        solution.subspan(offset, n_field_dofs));
}

LevelSetGlobalShiftCorrectionResult applyGlobalLevelSetShiftCorrection(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetVolumeOptions& volume_options,
    const LevelSetGlobalShiftCorrectionOptions& correction_options,
    std::span<const Real> coefficients,
    std::vector<Real>& corrected_coefficients)
{
    const auto collective = collectiveContext(level_set_dofs);
    requireQualifiedGlobalShiftLayout(mesh, level_set_dofs, collective);
    if (!(correction_options.volume_tolerance > 0.0) ||
        !std::isfinite(correction_options.volume_tolerance) ||
        !std::isfinite(correction_options.target_negative_volume)) {
        throw std::invalid_argument(
            "level-set global shift correction requires a finite target and positive finite volume tolerance");
    }
    if (correction_options.max_iterations <= 0) {
        throw std::invalid_argument("level-set global shift correction requires positive max_iterations");
    }
    if (!(correction_options.minimum_relative_volume_error >= 0.0) ||
        !std::isfinite(correction_options.minimum_relative_volume_error) ||
        !(correction_options.maximum_interface_displacement_fraction > 0.0) ||
        !(correction_options.maximum_interface_displacement_fraction <= 1.0) ||
        !std::isfinite(correction_options.maximum_interface_displacement_fraction)) {
        throw std::invalid_argument(
            "level-set global shift correction requires a finite nonnegative relative trigger and a displacement fraction in (0, 1]");
    }
    auto initial = computeLevelSetCutCellVolume(
        mesh,
        level_set_dofs,
        volume_options,
        coefficients);
    const Real target = correction_options.target_negative_volume;
    if (target < -correction_options.volume_tolerance ||
        target > initial.total_volume + correction_options.volume_tolerance) {
        throw std::invalid_argument(
            "level-set global shift correction target volume is outside the total volume range");
    }

    LevelSetGlobalShiftCorrectionResult result;
    result.target_negative_volume = target;
    result.initial_negative_volume = initial.negative_volume;
    result.initial_volume = initial;
    result.corrected_volume = initial;
    result.corrected_negative_volume = initial.negative_volume;
    result.volume_error = initial.negative_volume - target;
    result.trigger_volume_error = std::max(
        correction_options.volume_tolerance,
        correction_options.minimum_relative_volume_error *
            std::max(initial.total_volume, Real{0.0}));
    corrected_coefficients.assign(coefficients.begin(), coefficients.end());
    if (std::abs(result.volume_error) <= result.trigger_volume_error) {
        result.success = true;
        result.target_reached =
            std::abs(result.volume_error) <= correction_options.volume_tolerance;
        result.diagnostic = result.target_reached
                                ? std::string{}
                                : "level-set volume correction retained field below fallback trigger";
        return result;
    }
    result.correction_triggered = true;

    const auto* entity_map = level_set_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::invalid_argument("level-set global shift correction requires a scalar nodal field");
    }
    const auto [min_coeff, max_coeff] =
        vertexCoefficientRange(mesh, *entity_map, coefficients, collective);
    const Real pad = std::max(volume_options.tolerance * Real{10.0},
                              Real{1.0e-12});
    Real lower = volume_options.isovalue - max_coeff - pad;
    Real upper = volume_options.isovalue - min_coeff + pad;
    if (!(lower < upper)) {
        lower -= Real{1.0};
        upper += Real{1.0};
    }

    const Real maximum_topology_stable_shift =
        requireTopologyStableShiftBound(
            mesh,
            *entity_map,
            volume_options,
            coefficients,
            collective);
    result.maximum_topology_stable_shift = maximum_topology_stable_shift;
    bool bracket_was_clipped =
        lower < -maximum_topology_stable_shift ||
        upper > maximum_topology_stable_shift;
    lower = std::max(lower, -maximum_topology_stable_shift);
    upper = std::min(upper, maximum_topology_stable_shift);

    const auto zero_shift_metrics = shiftDisplacementMetrics(
        mesh,
        *entity_map,
        volume_options,
        coefficients,
        Real{0.0},
        collective);
    result.minimum_edge_length = zero_shift_metrics.minimum_edge_length;
    if (!(result.minimum_edge_length > Real{0.0}) ||
        !std::isfinite(result.minimum_edge_length)) {
        throw std::invalid_argument(
            "level-set global shift correction requires a positive finite "
            "minimum edge length for its displacement bound");
    }
    result.maximum_allowed_interface_displacement =
        correction_options.maximum_interface_displacement_fraction *
        result.minimum_edge_length;
    const Real bounded_lower = limitShiftByInterfaceDisplacement(
        mesh,
        *entity_map,
        volume_options,
        coefficients,
        lower,
        result.maximum_allowed_interface_displacement,
        collective);
    const Real bounded_upper = limitShiftByInterfaceDisplacement(
        mesh,
        *entity_map,
        volume_options,
        coefficients,
        upper,
        result.maximum_allowed_interface_displacement,
        collective);
    bracket_was_clipped = bracket_was_clipped ||
                          bounded_lower != lower || bounded_upper != upper;
    lower = bounded_lower;
    upper = bounded_upper;
    if (bracket_was_clipped) {
        const auto lower_volume = computeLevelSetCutCellVolume(
            mesh,
            level_set_dofs,
            volume_options,
            shiftedCoefficients(coefficients, lower));
        const auto upper_volume = computeLevelSetCutCellVolume(
            mesh,
            level_set_dofs,
            volume_options,
            shiftedCoefficients(coefficients, upper));
        const Real achievable_min = std::min(
            lower_volume.negative_volume, upper_volume.negative_volume);
        const Real achievable_max = std::max(
            lower_volume.negative_volume, upper_volume.negative_volume);
        result.limited_by_displacement_bound =
            target < achievable_min - correction_options.volume_tolerance ||
            target > achievable_max + correction_options.volume_tolerance;
    }

    Real best_shift = 0.0;
    Real best_error = std::abs(result.volume_error);
    LevelSetVolumeResult best_volume = initial;
    std::vector<Real> best_coefficients(coefficients.begin(), coefficients.end());

    for (int iter = 1; iter <= correction_options.max_iterations; ++iter) {
        const Real shift = Real{0.5} * (lower + upper);
        auto shifted = shiftedCoefficients(coefficients, shift);
        auto volume = computeLevelSetCutCellVolume(
            mesh,
            level_set_dofs,
            volume_options,
            shifted);
        const Real signed_error = volume.negative_volume - target;
        const Real abs_error = std::abs(signed_error);
        if (abs_error < best_error) {
            best_error = abs_error;
            best_shift = shift;
            best_volume = volume;
            best_coefficients = std::move(shifted);
        }

        result.iterations = iter;
        if (abs_error <= correction_options.volume_tolerance) {
            result.success = true;
            result.target_reached = true;
            result.correction_applied = std::abs(shift) > Real{0.0};
            result.applied_shift = shift;
            result.corrected_negative_volume = volume.negative_volume;
            result.volume_error = signed_error;
            result.corrected_volume = volume;
            const auto displacement = shiftDisplacementMetrics(
                mesh,
                *entity_map,
                volume_options,
                coefficients,
                shift,
                collective);
            result.max_interface_displacement =
                displacement.max_interface_displacement;
            result.max_contact_line_displacement =
                displacement.max_contact_line_displacement;
            result.contact_line_displacement_bound =
                result.max_contact_line_displacement;
            corrected_coefficients = std::move(best_coefficients);
            return result;
        }

        if (signed_error > 0.0) {
            lower = shift;
        } else {
            upper = shift;
        }
    }

    result.target_reached =
        best_error <= correction_options.volume_tolerance;
    // Reaching the displacement bound is an expected monitored fallback, not
    // an algorithm failure.  The result records that the exact target was not
    // reached while returning the best bounded correction.
    result.success = result.target_reached ||
                     result.limited_by_displacement_bound;
    result.correction_applied = std::abs(best_shift) > Real{0.0};
    result.applied_shift = best_shift;
    result.corrected_negative_volume = best_volume.negative_volume;
    result.volume_error = best_volume.negative_volume - target;
    result.corrected_volume = best_volume;
    const auto displacement = shiftDisplacementMetrics(
        mesh,
        *entity_map,
        volume_options,
        coefficients,
        best_shift,
        collective);
    result.max_interface_displacement = displacement.max_interface_displacement;
    result.max_contact_line_displacement =
        displacement.max_contact_line_displacement;
    result.contact_line_displacement_bound =
        result.max_contact_line_displacement;
    result.diagnostic = result.target_reached
                            ? std::string{}
                            : (result.limited_by_displacement_bound
                                   ? "level-set volume correction stopped at the interface-displacement/topology-stability bound"
                                   : "level-set global shift correction did not reach the requested volume tolerance");
    corrected_coefficients = std::move(best_coefficients);
    return result;
}

LevelSetGlobalShiftCorrectionResult applyGlobalLevelSetShiftCorrection(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetVolumeOptions& volume_options,
    const LevelSetGlobalShiftCorrectionOptions& correction_options,
    std::span<const Real> solution,
    std::vector<Real>& corrected_solution)
{
    requireQualifiedGlobalShiftField(system, level_set_field);
    if (volume_options.use_generated_interface_quadrature) {
        if (!(correction_options.volume_tolerance > 0.0) ||
            !std::isfinite(correction_options.volume_tolerance) ||
            !std::isfinite(correction_options.target_negative_volume)) {
            throw std::invalid_argument(
                "level-set global shift correction requires a finite target and positive finite volume tolerance");
        }
        if (correction_options.max_iterations <= 0) {
            throw std::invalid_argument(
                "level-set global shift correction requires positive max_iterations");
        }
        if (!(correction_options.minimum_relative_volume_error >= 0.0) ||
            !std::isfinite(correction_options.minimum_relative_volume_error) ||
            !(correction_options.maximum_interface_displacement_fraction > 0.0) ||
            !(correction_options.maximum_interface_displacement_fraction <= 1.0) ||
            !std::isfinite(correction_options.maximum_interface_displacement_fraction)) {
            throw std::invalid_argument(
                "level-set global shift correction requires a finite nonnegative relative trigger and a displacement fraction in (0, 1]");
        }

        const auto& field_dofs = system.fieldDofHandler(level_set_field);
        const auto collective = collectiveContext(field_dofs);
        const auto n_field_dofs =
            static_cast<std::size_t>(field_dofs.getNumDofs());
        const auto offset =
            static_cast<std::size_t>(system.fieldDofOffset(level_set_field));
        if (offset + n_field_dofs > solution.size()) {
            throw std::invalid_argument(
                "level-set global shift correction received an incompatible system solution span");
        }

        LevelSetGeneratedInterfaceLifecycle volume_lifecycle;
        auto initial = computeGeneratedInterfaceLevelSetVolume(
            system,
            level_set_field,
            volume_options,
            solution,
            volume_lifecycle);
        if (!initial.success) {
            LevelSetGlobalShiftCorrectionResult result;
            result.success = false;
            result.initial_volume = initial;
            result.corrected_volume = initial;
            result.target_negative_volume =
                correction_options.target_negative_volume;
            result.initial_negative_volume = initial.negative_volume;
            result.corrected_negative_volume = initial.negative_volume;
            result.volume_error =
                initial.negative_volume -
                correction_options.target_negative_volume;
            result.diagnostic = initial.diagnostic;
            accumulateGeneratedVolumeDiagnostics(result, initial);
            corrected_solution.assign(solution.begin(), solution.end());
            return result;
        }
        const Real target = correction_options.target_negative_volume;
        if (target < -correction_options.volume_tolerance ||
            target > initial.total_volume + correction_options.volume_tolerance) {
            throw std::invalid_argument(
                "level-set global shift correction target volume is outside the total volume range");
        }

        LevelSetGlobalShiftCorrectionResult result;
        result.target_negative_volume = target;
        result.initial_negative_volume = initial.negative_volume;
        result.initial_volume = initial;
        result.corrected_volume = initial;
        result.corrected_negative_volume = initial.negative_volume;
        result.volume_error = initial.negative_volume - target;
        result.trigger_volume_error = std::max(
            correction_options.volume_tolerance,
            correction_options.minimum_relative_volume_error *
                std::max(initial.total_volume, Real{0.0}));
        accumulateGeneratedVolumeDiagnostics(result, initial);
        corrected_solution.assign(solution.begin(), solution.end());
        if (std::abs(result.volume_error) <= result.trigger_volume_error) {
            result.success = true;
            result.target_reached =
                std::abs(result.volume_error) <= correction_options.volume_tolerance;
            result.diagnostic = result.target_reached
                                    ? std::string{}
                                    : "level-set volume correction retained field below fallback trigger";
            return result;
        }
        result.correction_triggered = true;

        const auto field_coefficients =
            solution.subspan(offset, n_field_dofs);
        const auto* entity_map = field_dofs.getEntityDofMap();
        if (entity_map == nullptr) {
            throw std::invalid_argument(
                "level-set global shift correction requires a scalar nodal field");
        }
        const auto [min_coeff, max_coeff] =
            coefficientRange(field_coefficients, collective);
        const Real pad = std::max(volume_options.tolerance * Real{10.0},
                                  Real{1.0e-12});
        Real lower = volume_options.isovalue - max_coeff - pad;
        Real upper = volume_options.isovalue - min_coeff + pad;
        if (!(lower < upper)) {
            lower -= Real{1.0};
            upper += Real{1.0};
        }

        const Real maximum_topology_stable_shift =
            requireTopologyStableShiftBound(
                system.meshAccess(),
                *entity_map,
                volume_options,
                field_coefficients,
                collective);
        result.maximum_topology_stable_shift = maximum_topology_stable_shift;
        bool bracket_was_clipped =
            lower < -maximum_topology_stable_shift ||
            upper > maximum_topology_stable_shift;
        lower = std::max(lower, -maximum_topology_stable_shift);
        upper = std::min(upper, maximum_topology_stable_shift);

        const auto zero_shift_metrics = shiftDisplacementMetrics(
            system.meshAccess(),
            *entity_map,
            volume_options,
            field_coefficients,
            Real{0.0},
            collective);
        result.minimum_edge_length = zero_shift_metrics.minimum_edge_length;
        if (!(result.minimum_edge_length > Real{0.0}) ||
            !std::isfinite(result.minimum_edge_length)) {
            throw std::invalid_argument(
                "level-set global shift correction requires a positive finite "
                "minimum edge length for its displacement bound");
        }
        result.maximum_allowed_interface_displacement =
            correction_options.maximum_interface_displacement_fraction *
            result.minimum_edge_length;
        const Real bounded_lower = limitShiftByInterfaceDisplacement(
            system.meshAccess(),
            *entity_map,
            volume_options,
            field_coefficients,
            lower,
            result.maximum_allowed_interface_displacement,
            collective);
        const Real bounded_upper = limitShiftByInterfaceDisplacement(
            system.meshAccess(),
            *entity_map,
            volume_options,
            field_coefficients,
            upper,
            result.maximum_allowed_interface_displacement,
            collective);
        bracket_was_clipped = bracket_was_clipped ||
                              bounded_lower != lower || bounded_upper != upper;
        lower = bounded_lower;
        upper = bounded_upper;
        if (bracket_was_clipped) {
            const auto lower_solution = shiftedSystemSolution(
                solution, offset, n_field_dofs, lower);
            const auto upper_solution = shiftedSystemSolution(
                solution, offset, n_field_dofs, upper);
            const auto lower_volume = computeGeneratedInterfaceLevelSetVolume(
                system,
                level_set_field,
                volume_options,
                lower_solution,
                volume_lifecycle);
            const auto upper_volume = computeGeneratedInterfaceLevelSetVolume(
                system,
                level_set_field,
                volume_options,
                upper_solution,
                volume_lifecycle);
            if (!lower_volume.success || !upper_volume.success) {
                result.success = false;
                result.diagnostic =
                    "level-set volume correction could not evaluate the displacement/topology-bounded bracket";
                return result;
            }
            const Real achievable_min = std::min(
                lower_volume.negative_volume,
                upper_volume.negative_volume);
            const Real achievable_max = std::max(
                lower_volume.negative_volume,
                upper_volume.negative_volume);
            result.limited_by_displacement_bound =
                target < achievable_min - correction_options.volume_tolerance ||
                target > achievable_max + correction_options.volume_tolerance;
        }

        Real best_shift = 0.0;
        Real best_error = std::abs(result.volume_error);
        LevelSetVolumeResult best_volume = initial;
        std::vector<Real> best_solution(solution.begin(), solution.end());

        for (int iter = 1; iter <= correction_options.max_iterations; ++iter) {
            const Real shift = Real{0.5} * (lower + upper);
            auto shifted = shiftedSystemSolution(
                solution,
                offset,
                n_field_dofs,
                shift);
            auto volume = computeGeneratedInterfaceLevelSetVolume(
                system,
                level_set_field,
                volume_options,
                shifted,
                volume_lifecycle);
            accumulateGeneratedVolumeDiagnostics(result, volume);
            if (!volume.success) {
                result.success = false;
                result.applied_shift = best_shift;
                result.corrected_negative_volume = best_volume.negative_volume;
                result.volume_error = best_volume.negative_volume - target;
                result.corrected_volume = best_volume;
                result.diagnostic = volume.diagnostic;
                corrected_solution = std::move(best_solution);
                return result;
            }
            const Real signed_error = volume.negative_volume - target;
            const Real abs_error = std::abs(signed_error);
            if (abs_error < best_error) {
                best_error = abs_error;
                best_shift = shift;
                best_volume = volume;
                best_solution = std::move(shifted);
            }

            result.iterations = iter;
            if (abs_error <= correction_options.volume_tolerance) {
                result.success = true;
                result.target_reached = true;
                result.correction_applied =
                    std::abs(best_shift) > Real{0.0};
                result.applied_shift = best_shift;
                result.corrected_negative_volume = best_volume.negative_volume;
                result.volume_error = best_volume.negative_volume - target;
                result.corrected_volume = best_volume;
                const auto displacement = shiftDisplacementMetrics(
                    system.meshAccess(),
                    *entity_map,
                    volume_options,
                    field_coefficients,
                    best_shift,
                    collective);
                result.max_interface_displacement =
                    displacement.max_interface_displacement;
                result.max_contact_line_displacement =
                    displacement.max_contact_line_displacement;
                result.contact_line_displacement_bound =
                    result.max_contact_line_displacement;
                corrected_solution = std::move(best_solution);
                return result;
            }

            if (signed_error > 0.0) {
                lower = shift;
            } else {
                upper = shift;
            }
        }

        result.target_reached =
            best_error <= correction_options.volume_tolerance;
        result.success = result.target_reached ||
                         result.limited_by_displacement_bound;
        result.correction_applied = std::abs(best_shift) > Real{0.0};
        result.applied_shift = best_shift;
        result.corrected_negative_volume = best_volume.negative_volume;
        result.volume_error = best_volume.negative_volume - target;
        result.corrected_volume = best_volume;
        const auto displacement = shiftDisplacementMetrics(
            system.meshAccess(),
            *entity_map,
            volume_options,
            field_coefficients,
            best_shift,
            collective);
        result.max_interface_displacement =
            displacement.max_interface_displacement;
        result.max_contact_line_displacement =
            displacement.max_contact_line_displacement;
        result.contact_line_displacement_bound =
            result.max_contact_line_displacement;
        result.diagnostic =
            result.target_reached
                ? std::string{}
                : (result.limited_by_displacement_bound
                       ? "level-set volume correction stopped at the interface-displacement/topology-stability bound"
                       : "level-set global shift correction did not reach the requested volume tolerance");
        corrected_solution = std::move(best_solution);
        return result;
    }

    const auto& field_dofs = system.fieldDofHandler(level_set_field);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(level_set_field));
    if (offset + n_field_dofs > solution.size()) {
        throw std::invalid_argument(
            "level-set global shift correction received an incompatible system solution span");
    }

    std::vector<Real> field_coefficients(
        solution.begin() + static_cast<std::ptrdiff_t>(offset),
        solution.begin() + static_cast<std::ptrdiff_t>(offset + n_field_dofs));
    std::vector<Real> corrected_field;
    auto result = applyGlobalLevelSetShiftCorrection(
        system.meshAccess(),
        field_dofs,
        volume_options,
        correction_options,
        field_coefficients,
        corrected_field);

    corrected_solution.assign(solution.begin(), solution.end());
    std::copy(corrected_field.begin(),
              corrected_field.end(),
              corrected_solution.begin() + static_cast<std::ptrdiff_t>(offset));
    return result;
}

} // namespace svmp::FE::level_set
