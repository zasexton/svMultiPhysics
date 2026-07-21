#include "LevelSet/LevelSetReinitialization.h"

#include "Dofs/EntityDofMap.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellTopology.h"
#endif

namespace svmp::FE::level_set {
namespace {

using interfaces::CutInterfaceDomainRequest;
using interfaces::CutInterfaceFragmentKind;
using interfaces::LevelSetCellCutInput;
using interfaces::LevelSetInterfaceSource;

void rejectDistributedProjectionReinitialization(
    const dofs::DofHandler& dof_handler)
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (initialized != 0 && finalized == 0 &&
        dof_handler.mpiComm() != MPI_COMM_NULL) {
        int communicator_size = 1;
        MPI_Comm_size(dof_handler.mpiComm(), &communicator_size);
        if (communicator_size > 1) {
            throw std::invalid_argument(
                "level-set signed-distance projection reinitialization is "
                "unsupported on MPI communicators with more than one rank: "
                "interface primitive construction and coefficient binding "
                "are currently rank-local");
        }
    }
#else
    (void)dof_handler;
#endif
}

struct SurfacePrimitive {
    CutInterfaceFragmentKind kind{CutInterfaceFragmentKind::Segment};
    GlobalIndex parent_cell{-1};
    std::vector<std::array<Real, 3>> points{};
};

[[nodiscard]] Real dot(const std::array<Real, 3>& a,
                       const std::array<Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
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

[[nodiscard]] Real norm(const std::array<Real, 3>& a) noexcept
{
    return std::sqrt(dot(a, a));
}

[[nodiscard]] Real distance(const std::array<Real, 3>& a,
                            const std::array<Real, 3>& b) noexcept
{
    return norm(sub(a, b));
}

[[nodiscard]] Real pointSegmentDistance(const std::array<Real, 3>& p,
                                        const std::array<Real, 3>& a,
                                        const std::array<Real, 3>& b) noexcept
{
    const auto ab = sub(b, a);
    const Real denom = dot(ab, ab);
    if (denom <= Real{0.0}) {
        return distance(p, a);
    }
    const Real t =
        std::clamp(dot(sub(p, a), ab) / denom, Real{0.0}, Real{1.0});
    return distance(p, add(a, scale(ab, t)));
}

[[nodiscard]] Real pointTriangleDistance(const std::array<Real, 3>& p,
                                         const std::array<Real, 3>& a,
                                         const std::array<Real, 3>& b,
                                         const std::array<Real, 3>& c) noexcept
{
    const auto ab = sub(b, a);
    const auto ac = sub(c, a);
    const auto ap = sub(p, a);
    const Real d1 = dot(ab, ap);
    const Real d2 = dot(ac, ap);
    if (d1 <= Real{0.0} && d2 <= Real{0.0}) {
        return distance(p, a);
    }

    const auto bp = sub(p, b);
    const Real d3 = dot(ab, bp);
    const Real d4 = dot(ac, bp);
    if (d3 >= Real{0.0} && d4 <= d3) {
        return distance(p, b);
    }

    const Real vc = d1 * d4 - d3 * d2;
    if (vc <= Real{0.0} && d1 >= Real{0.0} && d3 <= Real{0.0}) {
        const Real v = d1 / (d1 - d3);
        return distance(p, add(a, scale(ab, v)));
    }

    const auto cp = sub(p, c);
    const Real d5 = dot(ab, cp);
    const Real d6 = dot(ac, cp);
    if (d6 >= Real{0.0} && d5 <= d6) {
        return distance(p, c);
    }

    const Real vb = d5 * d2 - d1 * d6;
    if (vb <= Real{0.0} && d2 >= Real{0.0} && d6 <= Real{0.0}) {
        const Real w = d2 / (d2 - d6);
        return distance(p, add(a, scale(ac, w)));
    }

    const Real va = d3 * d6 - d5 * d4;
    if (va <= Real{0.0} && (d4 - d3) >= Real{0.0} &&
        (d5 - d6) >= Real{0.0}) {
        const Real w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return distance(p, add(b, scale(sub(c, b), w)));
    }

    const Real denom = Real{1.0} / (va + vb + vc);
    const Real v = vb * denom;
    const Real w = vc * denom;
    const auto closest = add(add(a, scale(ab, v)), scale(ac, w));
    return distance(p, closest);
}

[[nodiscard]] Real pointPrimitiveDistance(const std::array<Real, 3>& point,
                                          const SurfacePrimitive& primitive) noexcept
{
    if (primitive.kind == CutInterfaceFragmentKind::Segment &&
        primitive.points.size() >= 2u) {
        return pointSegmentDistance(point, primitive.points[0], primitive.points[1]);
    }
    if (primitive.kind == CutInterfaceFragmentKind::Polygon &&
        primitive.points.size() >= 3u) {
        Real best = std::numeric_limits<Real>::infinity();
        for (std::size_t i = 1u; i + 1u < primitive.points.size(); ++i) {
            best = std::min(best,
                            pointTriangleDistance(point,
                                                  primitive.points[0],
                                                  primitive.points[i],
                                                  primitive.points[i + 1u]));
        }
        return best;
    }
    Real best = std::numeric_limits<Real>::infinity();
    for (const auto& p : primitive.points) {
        best = std::min(best, distance(point, p));
    }
    return best;
}

[[nodiscard]] std::size_t cornerCount(ElementType type)
{
    switch (type) {
    case ElementType::Line2:
    case ElementType::Line3:
        return 2u;
    case ElementType::Triangle3:
    case ElementType::Triangle6:
        return 3u;
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return 4u;
    default:
        return 0u;
    }
}

[[nodiscard]] Real coefficientAtVertex(const dofs::EntityDofMap& entity_map,
                                       GlobalIndex vertex,
                                       std::span<const Real> coefficients)
{
    const auto dofs = entity_map.getVertexDofs(vertex);
    if (dofs.size() != 1u) {
        throw std::invalid_argument(
            "level-set signed-distance repair requires one scalar DOF per mesh vertex");
    }
    const auto dof = dofs.front();
    if (dof < 0 || static_cast<std::size_t>(dof) >= coefficients.size()) {
        throw std::invalid_argument(
            "level-set signed-distance repair found a vertex DOF outside the coefficient span");
    }
    return coefficients[static_cast<std::size_t>(dof)];
}

[[nodiscard]] std::span<const GlobalIndex> scalarVertexDofSpan(
    const dofs::EntityDofMap& entity_map,
    GlobalIndex vertex,
    std::size_t coefficient_count)
{
    const auto dofs = entity_map.getVertexDofs(vertex);
    if (dofs.empty()) {
        return dofs;
    }
    if (dofs.size() != 1u) {
        throw std::invalid_argument(
            "level-set signed-distance repair requires at most one scalar DOF per mesh vertex");
    }
    const auto dof = dofs.front();
    if (dof < 0 || static_cast<std::size_t>(dof) >= coefficient_count) {
        throw std::invalid_argument(
            "level-set signed-distance repair found a vertex DOF outside the coefficient span");
    }
    return dofs;
}

[[nodiscard]] Real nearestDistanceToInterface(
    const std::array<Real, 3>& point,
    const std::vector<SurfacePrimitive>& primitives)
{
    Real best = std::numeric_limits<Real>::infinity();
    for (const auto& primitive : primitives) {
        best = std::min(best, pointPrimitiveDistance(point, primitive));
    }
    return best;
}

[[nodiscard]] Real distanceToPrimitiveSupportingGeometry(
    const std::array<Real, 3>& point,
    const SurfacePrimitive& primitive,
    int dimension) noexcept
{
    if (primitive.points.empty()) {
        return std::numeric_limits<Real>::infinity();
    }
    if (primitive.points.size() == 1u || dimension <= 1) {
        return distance(point, primitive.points.front());
    }
    if (dimension == 2 || primitive.points.size() == 2u) {
        const auto direction = sub(primitive.points[1], primitive.points[0]);
        const Real length_squared = dot(direction, direction);
        if (!(length_squared > Real{0.0})) {
            return distance(point, primitive.points[0]);
        }
        const auto offset = sub(point, primitive.points[0]);
        const Real projection = dot(offset, direction) / length_squared;
        return norm(sub(offset, scale(direction, projection)));
    }

    const auto first = primitive.points.front();
    for (std::size_t i = 1u; i + 1u < primitive.points.size(); ++i) {
        const auto a = sub(primitive.points[i], first);
        const auto b = sub(primitive.points[i + 1u], first);
        const auto normal = cross(a, b);
        const Real normal_length = norm(normal);
        if (normal_length > Real{0.0}) {
            return std::abs(dot(sub(point, first), normal)) / normal_length;
        }
    }
    return pointPrimitiveDistance(point, primitive);
}

struct LinearInterfacePrimitiveSet {
    std::vector<SurfacePrimitive> primitives{};
    std::vector<GlobalIndex> cut_cell_ids{};
    std::size_t cut_cells{0u};
};

[[nodiscard]] LinearInterfacePrimitiveSet buildLinearInterfacePrimitives(
    const assembly::IMeshAccess& mesh,
    const dofs::EntityDofMap& entity_map,
    Real tolerance,
    std::span<const Real> coefficients)
{
    CutInterfaceDomainRequest request{};
    request.source = LevelSetInterfaceSource::fromField(FieldId{0});
    request.interface_marker = 0;
    request.tolerance = tolerance;
    request.quadrature_order = 1;

    LinearInterfacePrimitiveSet output;
    std::vector<GlobalIndex> cell_nodes;
    std::vector<std::array<Real, 3>> cell_coordinates;
    mesh.forEachCell([&](GlobalIndex cell_id) {
        const auto type = mesh.getCellType(cell_id);
        const std::size_t count = cornerCount(type);
        if (count == 0u) {
            return;
        }

        mesh.getCellNodes(cell_id, cell_nodes);
        mesh.getCellCoordinates(cell_id, cell_coordinates);
        if (cell_nodes.size() < count || cell_coordinates.size() < count) {
            return;
        }

        LevelSetCellCutInput input{};
        input.parent_cell = cell_id;
        input.element_type = type;
        input.node_coordinates.assign(cell_coordinates.begin(),
                                      cell_coordinates.begin() +
                                          static_cast<std::ptrdiff_t>(count));
        input.level_set_values.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            input.level_set_values.push_back(
                coefficientAtVertex(entity_map,
                                    cell_nodes[i],
                                    coefficients));
        }

        interfaces::LevelSetCellCutResult cut_result;
        if (mesh.dimension() == 2) {
            cut_result = interfaces::cutLinearLevelSetCell2D(request, input);
        } else if (mesh.dimension() == 3) {
            cut_result = interfaces::cutLinearLevelSetCell3D(request, input);
        } else {
            return;
        }

        bool added_cell_fragment = false;
        for (const auto& fragment : cut_result.fragments) {
            if (!fragment.active()) {
                continue;
            }
            SurfacePrimitive primitive;
            primitive.kind = fragment.kind;
            primitive.parent_cell = cell_id;
            primitive.points.reserve(fragment.vertices.size());
            for (const auto& vertex : fragment.vertices) {
                primitive.points.push_back(vertex.point);
            }
            output.primitives.push_back(std::move(primitive));
            added_cell_fragment = true;
        }
        if (added_cell_fragment) {
            output.cut_cell_ids.push_back(cell_id);
            ++output.cut_cells;
        }
    });
    return output;
}

class DisjointSet {
public:
    explicit DisjointSet(std::size_t size) : parent_(size), rank_(size, 0u)
    {
        for (std::size_t i = 0; i < size; ++i) {
            parent_[i] = i;
        }
    }

    [[nodiscard]] std::size_t find(std::size_t value)
    {
        if (parent_[value] != value) {
            parent_[value] = find(parent_[value]);
        }
        return parent_[value];
    }

    void unite(std::size_t a, std::size_t b)
    {
        a = find(a);
        b = find(b);
        if (a == b) {
            return;
        }
        if (rank_[a] < rank_[b]) {
            std::swap(a, b);
        }
        parent_[b] = a;
        if (rank_[a] == rank_[b]) {
            ++rank_[a];
        }
    }

private:
    std::vector<std::size_t> parent_{};
    std::vector<unsigned char> rank_{};
};

[[nodiscard]] std::vector<std::array<std::size_t, 2>> cornerEdges(
    ElementType type)
{
    switch (type) {
    case ElementType::Line2:
    case ElementType::Line3:
        return {{{0u, 1u}}};
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
    default:
        return {};
    }
}

struct EdgeZeroCrossing {
    GlobalIndex dof_a{-1};
    GlobalIndex dof_b{-1};
    std::array<Real, 3> point_a{};
    std::array<Real, 3> point_b{};
    Real original_t{0.0};
};

struct ZeroSetDisplacementEvaluation {
    bool topology_preserved{true};
    Real max_displacement{0.0};
    Real l2_displacement{0.0};
    std::size_t samples{0u};
};

[[nodiscard]] std::vector<EdgeZeroCrossing> collectOriginalZeroCrossings(
    const assembly::IMeshAccess& mesh,
    const dofs::EntityDofMap& entity_map,
    const LinearInterfacePrimitiveSet& primitive_set,
    Real tolerance,
    std::span<const Real> coefficients)
{
    std::vector<EdgeZeroCrossing> crossings;
    std::vector<GlobalIndex> nodes;
    std::vector<std::array<Real, 3>> points;
    for (const auto cell : primitive_set.cut_cell_ids) {
        mesh.getCellNodes(cell, nodes);
        mesh.getCellCoordinates(cell, points);
        for (const auto edge : cornerEdges(mesh.getCellType(cell))) {
            if (edge[0] >= nodes.size() || edge[1] >= nodes.size() ||
                edge[0] >= points.size() || edge[1] >= points.size()) {
                continue;
            }
            const auto dofs_a = entity_map.getVertexDofs(nodes[edge[0]]);
            const auto dofs_b = entity_map.getVertexDofs(nodes[edge[1]]);
            if (dofs_a.size() != 1u || dofs_b.size() != 1u) {
                continue;
            }
            const auto ia = static_cast<std::size_t>(dofs_a.front());
            const auto ib = static_cast<std::size_t>(dofs_b.front());
            if (ia >= coefficients.size() || ib >= coefficients.size()) {
                continue;
            }
            const Real a = coefficients[ia];
            const Real b = coefficients[ib];
            if (!((a < -tolerance && b > tolerance) ||
                  (a > tolerance && b < -tolerance))) {
                continue;
            }
            const Real denom = a - b;
            if (std::abs(denom) <= tolerance) {
                continue;
            }
            crossings.push_back(EdgeZeroCrossing{
                .dof_a = dofs_a.front(),
                .dof_b = dofs_b.front(),
                .point_a = points[edge[0]],
                .point_b = points[edge[1]],
                .original_t = std::clamp(a / denom, Real{0.0}, Real{1.0}),
            });
        }
    }
    std::sort(crossings.begin(), crossings.end(), [](const auto& a, const auto& b) {
        const auto a0 = std::min(a.dof_a, a.dof_b);
        const auto a1 = std::max(a.dof_a, a.dof_b);
        const auto b0 = std::min(b.dof_a, b.dof_b);
        const auto b1 = std::max(b.dof_a, b.dof_b);
        return std::pair{a0, a1} < std::pair{b0, b1};
    });
    crossings.erase(
        std::unique(crossings.begin(), crossings.end(), [](const auto& a, const auto& b) {
            return std::min(a.dof_a, a.dof_b) == std::min(b.dof_a, b.dof_b) &&
                   std::max(a.dof_a, a.dof_b) == std::max(b.dof_a, b.dof_b);
        }),
        crossings.end());
    return crossings;
}

[[nodiscard]] ZeroSetDisplacementEvaluation evaluateZeroSetDisplacement(
    std::span<const EdgeZeroCrossing> crossings,
    std::span<const Real> coefficients,
    Real tolerance)
{
    ZeroSetDisplacementEvaluation evaluation;
    Real displacement_squared_sum = 0.0;
    for (const auto& crossing : crossings) {
        const auto ia = static_cast<std::size_t>(crossing.dof_a);
        const auto ib = static_cast<std::size_t>(crossing.dof_b);
        if (ia >= coefficients.size() || ib >= coefficients.size()) {
            evaluation.topology_preserved = false;
            return evaluation;
        }
        const Real a = coefficients[ia];
        const Real b = coefficients[ib];
        const Real denominator = a - b;
        if (!std::isfinite(a) || !std::isfinite(b) ||
            !std::isfinite(denominator) ||
            std::abs(denominator) <= tolerance ||
            !((a < Real{0.0} && b > Real{0.0}) ||
              (a > Real{0.0} && b < Real{0.0}))) {
            evaluation.topology_preserved = false;
            return evaluation;
        }
        const Real repaired_t =
            std::clamp(a / denominator, Real{0.0}, Real{1.0});
        const Real displacement =
            std::abs(repaired_t - crossing.original_t) *
            distance(crossing.point_a, crossing.point_b);
        evaluation.max_displacement =
            std::max(evaluation.max_displacement, displacement);
        displacement_squared_sum += displacement * displacement;
        ++evaluation.samples;
    }
    if (evaluation.samples > 0u) {
        evaluation.l2_displacement =
            std::sqrt(displacement_squared_sum /
                      static_cast<Real>(evaluation.samples));
    }
    return evaluation;
}

template <typename ForEachDofPoint>
[[nodiscard]] LevelSetSignedDistanceRepairResult repairSignedDistanceCoefficientsFromPrimitives(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& field_dofs,
    const dofs::EntityDofMap& entity_map,
    const LevelSetReinitializationOptions& options,
    std::span<const Real> input_coefficients,
    std::vector<Real>& repaired_coefficients,
    const LinearInterfacePrimitiveSet& primitive_set,
    ForEachDofPoint&& for_each_dof_point)
{
    LevelSetSignedDistanceRepairResult result;
    result.method = LevelSetReinitializationMethod::Projection;
    result.interface_fragments = primitive_set.primitives.size();
    result.cut_cells = primitive_set.cut_cells;
    if (primitive_set.primitives.empty()) {
        result.success = false;
        result.diagnostic = "level-set signed-distance repair found no active interface fragments";
        return result;
    }

    const auto expected = input_coefficients.size();
    std::vector<unsigned char> bound(expected, 0u);
    std::vector<std::array<Real, 3>> dof_points(expected);
    const auto collect_dof_point = [&](GlobalIndex dof,
                                       const std::array<Real, 3>& x) {
        if (dof < 0 || static_cast<std::size_t>(dof) >= expected) {
            throw std::invalid_argument(
                "level-set signed-distance repair found a cell DOF outside the coefficient span");
        }
        const auto dof_index = static_cast<std::size_t>(dof);
        if (bound[dof_index] != 0u) {
            return;
        }
        dof_points[dof_index] = x;
        bound[dof_index] = 1u;
    };

    for_each_dof_point(collect_dof_point);

    const auto unrepaired =
        static_cast<std::size_t>(std::count(bound.begin(),
                                            bound.end(),
                                            static_cast<unsigned char>(0u)));
    if (unrepaired != 0u) {
        result.success = false;
        result.diagnostic =
            "level-set signed-distance repair left " +
            std::to_string(unrepaired) +
            " coefficient(s) without an entity-aware mesh-node binding";
        return result;
    }

    // In cut cells, use the supporting line/plane of the local interface
    // fragment.  Extending that geometry past the clipped cell/wall endpoint
    // avoids the radial-distance artifact that otherwise rotates normals at a
    // contact point.  Away from cut cells, retain the true nearest finite-
    // primitive distance.
    std::vector<Real> cut_support_distance(
        expected, std::numeric_limits<Real>::infinity());
    for (const auto cell : primitive_set.cut_cell_ids) {
        const auto cell_dofs = field_dofs.getCellDofs(cell);
        for (const auto dof : cell_dofs) {
            if (dof < 0 || static_cast<std::size_t>(dof) >= expected) {
                throw std::invalid_argument(
                    "level-set signed-distance repair found a cut-cell DOF outside the coefficient span");
            }
            auto& local_distance =
                cut_support_distance[static_cast<std::size_t>(dof)];
            for (const auto& primitive : primitive_set.primitives) {
                if (primitive.parent_cell != cell) {
                    continue;
                }
                local_distance = std::min(
                    local_distance,
                    distanceToPrimitiveSupportingGeometry(
                        dof_points[static_cast<std::size_t>(dof)],
                        primitive,
                        mesh.dimension()));
            }
        }
    }

    std::vector<Real> signed_distance_target(expected, 0.0);
    std::vector<Real> distances(expected, 0.0);
    for (std::size_t i = 0; i < expected; ++i) {
        const Real finite_distance = nearestDistanceToInterface(
            dof_points[i], primitive_set.primitives);
        const Real d = std::isfinite(cut_support_distance[i])
                           ? cut_support_distance[i]
                           : finite_distance;
        if (!std::isfinite(d)) {
            throw std::runtime_error(
                "level-set signed-distance repair produced a non-finite distance");
        }
        distances[i] = d;
        result.max_distance = std::max(result.max_distance, d);
        if (input_coefficients[i] > options.signed_distance_tolerance) {
            signed_distance_target[i] = d;
        } else if (input_coefficients[i] < -options.signed_distance_tolerance) {
            signed_distance_target[i] = -d;
        }
    }

    // Linear nodal cells can move toward the local signed-distance target while
    // a geometric line search below explicitly bounds movement of every
    // original cut-edge crossing.  High-order cells need a stronger invariant:
    // a common positive multiplier on all DOFs in their connected cut patch
    // preserves the complete polynomial zero set, including roots that are not
    // visible on corner edges.
    DisjointSet components(expected);
    std::vector<unsigned char> in_cut_patch(expected, 0u);
    for (const auto cell : primitive_set.cut_cell_ids) {
        const auto cell_dofs = field_dofs.getCellDofs(cell);
        std::optional<std::size_t> first;
        for (const auto dof : cell_dofs) {
            if (dof < 0 || static_cast<std::size_t>(dof) >= expected) {
                throw std::invalid_argument(
                    "level-set signed-distance repair found a cut-cell DOF outside the coefficient span");
            }
            const auto index = static_cast<std::size_t>(dof);
            in_cut_patch[index] = 1u;
            if (first.has_value()) {
                components.unite(*first, index);
            } else {
                first = index;
            }
        }
    }

    std::map<std::size_t, bool> component_requires_common_scale;
    for (std::size_t i = 0; i < expected; ++i) {
        if (in_cut_patch[i] != 0u) {
            component_requires_common_scale.emplace(components.find(i), false);
        }
    }
    for (const auto cell : primitive_set.cut_cell_ids) {
        const auto cell_dofs = field_dofs.getCellDofs(cell);
        const bool high_order =
            cell_dofs.size() > cornerCount(mesh.getCellType(cell));
        if (!high_order) {
            continue;
        }
        for (const auto dof : cell_dofs) {
            component_requires_common_scale[components.find(
                static_cast<std::size_t>(dof))] = true;
        }
    }

    std::map<std::size_t, std::pair<Real, Real>> scale_sums;
    for (std::size_t i = 0; i < expected; ++i) {
        if (in_cut_patch[i] == 0u) {
            continue;
        }
        auto& sums = scale_sums[components.find(i)];
        sums.first += input_coefficients[i] * signed_distance_target[i];
        sums.second += input_coefficients[i] * input_coefficients[i];
    }
    std::map<std::size_t, Real> component_target_scale;
    for (const auto& [root, sums] : scale_sums) {
        Real fitted = Real{1.0};
        if (sums.second > options.signed_distance_tolerance *
                              options.signed_distance_tolerance) {
            fitted = sums.first / sums.second;
        }
        if (!std::isfinite(fitted) || fitted <= Real{0.0}) {
            fitted = Real{1.0};
        }
        component_target_scale[root] = fitted;
    }

    std::vector<Real> target(expected, 0.0);
    std::vector<unsigned char> update_enabled(expected, 1u);
    for (std::size_t i = 0; i < expected; ++i) {
        if (in_cut_patch[i] != 0u) {
            const auto root = components.find(i);
            target[i] = component_requires_common_scale.at(root)
                            ? component_target_scale.at(root) *
                                  input_coefficients[i]
                            : signed_distance_target[i];
        } else {
            target[i] = signed_distance_target[i];
            if (options.interface_band_width > Real{0.0} &&
                distances[i] > options.interface_band_width) {
                update_enabled[i] = 0u;
                target[i] = input_coefficients[i];
                ++result.preserved_dofs;
            }
        }
    }

    const Real relaxation = std::clamp(
        options.pseudo_time_step_scale, Real{0.0}, Real{1.0});
    std::vector<Real> unconstrained(input_coefficients.begin(),
                                    input_coefficients.end());
    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        Real max_remaining = 0.0;
        for (std::size_t i = 0; i < expected; ++i) {
            if (update_enabled[i] == 0u) {
                continue;
            }
            unconstrained[i] +=
                relaxation * (target[i] - unconstrained[i]);
            max_remaining = std::max(
                max_remaining,
                std::abs(target[i] - unconstrained[i]));
        }
        result.iterations = iteration;
        result.max_iteration_residual = max_remaining;
        if (max_remaining <= options.signed_distance_tolerance) {
            break;
        }
    }

    const auto crossings = collectOriginalZeroCrossings(
        mesh,
        entity_map,
        primitive_set,
        options.signed_distance_tolerance,
        input_coefficients);
    const Real displacement_gate = std::max(
        options.max_zero_set_displacement,
        Real{32.0} * std::numeric_limits<Real>::epsilon());

    repaired_coefficients = unconstrained;
    auto displacement = evaluateZeroSetDisplacement(
        crossings,
        repaired_coefficients,
        options.signed_distance_tolerance);
    if (!displacement.topology_preserved ||
        displacement.max_displacement > displacement_gate) {
        // Along a convex coefficient path every individual linear-edge root is
        // a monotone fractional-linear function until topology changes.  A
        // scalar bisection therefore finds the largest admissible redistance
        // update without exceeding the user-visible geometric guard.
        Real lower = Real{0.0};
        Real upper = Real{1.0};
        std::vector<Real> candidate(expected, 0.0);
        for (int iteration = 0; iteration < 64; ++iteration) {
            const Real fraction = Real{0.5} * (lower + upper);
            for (std::size_t i = 0; i < expected; ++i) {
                candidate[i] = input_coefficients[i] +
                               fraction *
                                   (unconstrained[i] - input_coefficients[i]);
            }
            const auto trial = evaluateZeroSetDisplacement(
                crossings,
                candidate,
                options.signed_distance_tolerance);
            if (trial.topology_preserved &&
                trial.max_displacement <= displacement_gate) {
                lower = fraction;
            } else {
                upper = fraction;
            }
        }
        for (std::size_t i = 0; i < expected; ++i) {
            repaired_coefficients[i] = input_coefficients[i] +
                                       lower *
                                           (unconstrained[i] -
                                            input_coefficients[i]);
        }
        displacement = evaluateZeroSetDisplacement(
            crossings,
            repaired_coefficients,
            options.signed_distance_tolerance);
    }

    result.max_iteration_residual = 0.0;
    for (std::size_t i = 0; i < expected; ++i) {
        if (update_enabled[i] == 0u) {
            continue;
        }
        const Real abs_update =
            std::abs(repaired_coefficients[i] - input_coefficients[i]);
        result.max_abs_update = std::max(result.max_abs_update, abs_update);
        result.max_signed_distance_error = std::max(
            result.max_signed_distance_error,
            std::abs(repaired_coefficients[i] - signed_distance_target[i]));
        result.max_iteration_residual = std::max(
            result.max_iteration_residual,
            std::abs(repaired_coefficients[i] - target[i]));
        ++result.repaired_dofs;
    }

    if (!displacement.topology_preserved) {
        result.success = false;
        result.diagnostic =
            "level-set signed-distance repair changed interface topology on a cut edge";
        return result;
    }
    result.max_interface_displacement = displacement.max_displacement;
    result.l2_interface_displacement = displacement.l2_displacement;
    result.interface_displacement_samples = displacement.samples;
    result.zero_set_bound_satisfied =
        displacement.max_displacement <= displacement_gate;
    if (!result.zero_set_bound_satisfied) {
        result.success = false;
        result.diagnostic =
            "level-set signed-distance repair exceeded max_zero_set_displacement";
        return result;
    }
    result.converged =
        result.max_iteration_residual <= options.signed_distance_tolerance &&
        result.max_signed_distance_error <= options.signed_distance_tolerance;
    result.success = true;
    if (!result.converged) {
        result.diagnostic =
            "level-set signed-distance repair did not reach the signed-distance tolerance; partial repair must not be applied in production";
    }
    return result;
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
struct GeometryOrderInfo {
    int order{1};
    svmp::CellTopology::HighOrderKind kind{
        svmp::CellTopology::HighOrderKind::Lagrange};
};

[[nodiscard]] GeometryOrderInfo inferGeometryOrder(
    svmp::CellFamily family,
    int declared_order,
    int num_corners,
    std::size_t node_count)
{
    GeometryOrderInfo info;
    info.order = std::max(1, declared_order);
    const int corners = std::max(0, num_corners);
    if (node_count <= static_cast<std::size_t>(corners)) {
        return info;
    }
    if (family == svmp::CellFamily::Line) {
        if (node_count >= 2u) {
            info.order = static_cast<int>(node_count) - 1;
        }
        return info;
    }
    const int p_lag =
        svmp::CellTopology::infer_lagrange_order(family, node_count);
    const int p_ser =
        svmp::CellTopology::infer_serendipity_order(family, node_count);
    if (p_lag > 0 &&
        (declared_order <= 1 || p_lag == declared_order ||
         p_ser != declared_order)) {
        info.order = p_lag;
        info.kind = svmp::CellTopology::HighOrderKind::Lagrange;
    } else if (p_ser > 0) {
        info.order = p_ser;
        info.kind = svmp::CellTopology::HighOrderKind::Serendipity;
    } else if (p_lag > 0) {
        info.order = p_lag;
        info.kind = svmp::CellTopology::HighOrderKind::Lagrange;
    }
    return info;
}

[[nodiscard]] int topologicalDimension(svmp::CellFamily family) noexcept
{
    switch (family) {
    case svmp::CellFamily::Line:
        return 1;
    case svmp::CellFamily::Triangle:
    case svmp::CellFamily::Quad:
    case svmp::CellFamily::Polygon:
        return 2;
    case svmp::CellFamily::Tetra:
    case svmp::CellFamily::Hex:
    case svmp::CellFamily::Wedge:
    case svmp::CellFamily::Pyramid:
    case svmp::CellFamily::Polyhedron:
        return 3;
    default:
        return 0;
    }
}

[[nodiscard]] std::vector<svmp::index_t> faceInteriorGeometryNodes(
    const svmp::MeshBase& mesh,
    svmp::index_t face)
{
    auto [face_nodes, n_face_nodes] = mesh.face_vertices_span(face);
    std::vector<svmp::index_t> interior;
    if (face_nodes == nullptr || n_face_nodes == 0u) {
        return interior;
    }

    svmp::CellShape shape{};
    shape.num_corners = static_cast<int>(n_face_nodes);
    const auto& face_shapes = mesh.face_shapes();
    if (static_cast<std::size_t>(face) < face_shapes.size()) {
        shape = face_shapes[static_cast<std::size_t>(face)];
    }
    if (topologicalDimension(shape.family) != 2) {
        return interior;
    }

    const auto info =
        inferGeometryOrder(shape.family, shape.order, shape.num_corners,
                           n_face_nodes);
    std::vector<bool> on_boundary(n_face_nodes, false);
    const auto boundary =
        svmp::CellTopology::get_oriented_boundary_faces_view(shape.family);
    if (boundary.face_count <= 0) {
        const int corners =
            std::min<int>(std::max(0, shape.num_corners),
                          static_cast<int>(n_face_nodes));
        for (int i = 0; i < corners; ++i) {
            on_boundary[static_cast<std::size_t>(i)] = true;
        }
    } else {
        for (int local_face = 0; local_face < boundary.face_count; ++local_face) {
            for (const auto local_node :
                 svmp::CellTopology::high_order_face_local_nodes(
                     shape.family, info.order, local_face, info.kind)) {
                const auto idx = static_cast<std::size_t>(local_node);
                if (idx < n_face_nodes) {
                    on_boundary[idx] = true;
                }
            }
        }
    }

    for (std::size_t i = 0; i < n_face_nodes; ++i) {
        if (!on_boundary[i]) {
            interior.push_back(face_nodes[i]);
        }
    }
    return interior;
}

template <typename Callback>
void forEachNativeMeshScalarDofPoint(
    const svmp::MeshBase& mesh,
    const assembly::IMeshAccess& coordinate_access,
    const dofs::EntityDofMap& entity_map,
    std::size_t coefficient_count,
    Callback&& callback)
{
    std::vector<unsigned char> bound(coefficient_count, 0u);
    const auto bind_point = [&](GlobalIndex dof,
                                const std::array<Real, 3>& point) {
        if (dof < 0 || static_cast<std::size_t>(dof) >= coefficient_count) {
            throw std::invalid_argument(
                "level-set signed-distance repair found an entity DOF outside the coefficient span");
        }
        const auto sdof = static_cast<std::size_t>(dof);
        if (bound[sdof] != 0u) {
            return;
        }
        callback(dof, point);
        bound[sdof] = 1u;
    };
    const auto bind = [&](svmp::index_t geometry_node, GlobalIndex dof) {
        if (geometry_node < 0 ||
            static_cast<std::size_t>(geometry_node) >=
                static_cast<std::size_t>(mesh.n_vertices())) {
            throw std::invalid_argument(
                "level-set signed-distance repair found a mesh geometry node outside the mesh");
        }
        bind_point(dof, coordinate_access.getNodeCoordinates(geometry_node));
    };

    const auto n_vertices = static_cast<GlobalIndex>(mesh.n_vertices());
    if (entity_map.numVertices() < n_vertices) {
        throw std::invalid_argument(
            "level-set signed-distance repair field does not cover every mesh vertex");
    }
    for (GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
        const auto dofs = entity_map.getVertexDofs(vertex);
        if (dofs.empty()) {
            continue;
        }
        if (dofs.size() != 1u) {
            throw std::invalid_argument(
                "level-set signed-distance repair requires scalar vertex DOFs");
        }
        bind(static_cast<svmp::index_t>(vertex), dofs.front());
    }

    std::map<std::pair<svmp::index_t, svmp::index_t>, svmp::index_t>
        edge_by_vertices;
    auto make_edge_key = [](svmp::index_t a, svmp::index_t b) {
        if (b < a) {
            std::swap(a, b);
        }
        return std::pair<svmp::index_t, svmp::index_t>{a, b};
    };
    for (svmp::index_t edge = 0;
         edge < static_cast<svmp::index_t>(mesh.n_edges());
         ++edge) {
        const auto vertices = mesh.edge_vertices(edge);
        edge_by_vertices.emplace(make_edge_key(vertices[0], vertices[1]), edge);
    }
    const auto vertex_gid = [&](svmp::index_t vertex) -> svmp::gid_t {
        const auto sv = static_cast<std::size_t>(vertex);
        if (vertex >= 0 && sv < mesh.vertex_gids().size()) {
            return mesh.vertex_gids()[sv];
        }
        return static_cast<svmp::gid_t>(vertex);
    };
    const auto canonical_edge_dof_endpoints =
        [&](std::array<svmp::index_t, 2> endpoints) {
        if (vertex_gid(endpoints[1]) < vertex_gid(endpoints[0])) {
            std::swap(endpoints[0], endpoints[1]);
        }
        return endpoints;
    };
    const auto interpolate_coordinates =
        [&](svmp::index_t endpoint_a,
            svmp::index_t endpoint_b,
            Real t) -> std::array<Real, 3> {
        const auto a = coordinate_access.getNodeCoordinates(endpoint_a);
        const auto b = coordinate_access.getNodeCoordinates(endpoint_b);
        return add(scale(a, Real{1} - t), scale(b, t));
    };

    auto bind_edge_interior = [&](svmp::index_t cell,
                                  int local_edge,
                                  svmp::index_t endpoint_a,
                                  svmp::index_t endpoint_b) {
        const auto edge_it =
            edge_by_vertices.find(make_edge_key(endpoint_a, endpoint_b));
        if (edge_it == edge_by_vertices.end()) {
            return;
        }
        const auto edge = edge_it->second;
        const auto edge_geometry = mesh.cell_edge_geometry_dofs(cell, local_edge);
        const auto edge_dofs = entity_map.getEdgeDofs(edge);
        if (edge_dofs.empty()) {
            return;
        }
        if (edge_geometry.size() <= 2u) {
            const auto endpoints =
                canonical_edge_dof_endpoints(mesh.edge_vertices(edge));
            for (std::size_t j = 0; j < edge_dofs.size(); ++j) {
                const Real t =
                    static_cast<Real>(j + 1u) /
                    static_cast<Real>(edge_dofs.size() + 1u);
                bind_point(edge_dofs[j],
                           interpolate_coordinates(endpoints[0],
                                                   endpoints[1],
                                                   t));
            }
            return;
        }
        const auto canonical = mesh.edge_vertices(edge);
        const bool forward =
            edge_geometry.front() == canonical[0] &&
            edge_geometry.back() == canonical[1];
        const bool reverse =
            edge_geometry.front() == canonical[1] &&
            edge_geometry.back() == canonical[0];
        if (!forward && !reverse) {
            throw std::invalid_argument(
                "level-set signed-distance repair found high-order edge geometry inconsistent with mesh topology");
        }

        const auto interior_count = edge_geometry.size() - 2u;
        if (edge_dofs.size() != interior_count) {
            throw std::invalid_argument(
                "level-set signed-distance repair edge DOF count does not match high-order mesh edge nodes");
        }
        for (std::size_t j = 0; j < interior_count; ++j) {
            const auto geometry_index =
                forward ? (j + 1u) : (edge_geometry.size() - 2u - j);
            bind(edge_geometry[geometry_index], edge_dofs[j]);
        }
    };

    for (svmp::index_t cell = 0;
         cell < static_cast<svmp::index_t>(mesh.n_cells());
         ++cell) {
        auto [cell_vertices, n_cell_vertices] = mesh.cell_vertices_span(cell);
        if (cell_vertices == nullptr || n_cell_vertices == 0u) {
            continue;
        }
        const auto& shape = mesh.cell_shape(cell);
        if (shape.family == svmp::CellFamily::Polygon) {
            const int corner_count =
                shape.num_corners > 0
                    ? std::min<int>(shape.num_corners,
                                    static_cast<int>(n_cell_vertices))
                    : static_cast<int>(n_cell_vertices);
            if (corner_count >= 2) {
                const auto edges =
                    svmp::CellTopology::get_polygon_edges_view(corner_count);
                for (int local_edge = 0; local_edge < edges.edge_count;
                     ++local_edge) {
                    const auto local_a = edges.pairs_flat[2 * local_edge];
                    const auto local_b = edges.pairs_flat[2 * local_edge + 1];
                    if (local_a < 0 || local_b < 0 ||
                        static_cast<std::size_t>(local_a) >= n_cell_vertices ||
                        static_cast<std::size_t>(local_b) >= n_cell_vertices) {
                        continue;
                    }
                    bind_edge_interior(
                        cell,
                        local_edge,
                        cell_vertices[static_cast<std::size_t>(local_a)],
                        cell_vertices[static_cast<std::size_t>(local_b)]);
                }
            }
        } else {
            const auto edges = svmp::CellTopology::get_edges_view(shape.family);
            for (int local_edge = 0; local_edge < edges.edge_count;
                 ++local_edge) {
                const auto local_a = edges.pairs_flat[2 * local_edge];
                const auto local_b = edges.pairs_flat[2 * local_edge + 1];
                if (local_a < 0 || local_b < 0 ||
                    static_cast<std::size_t>(local_a) >= n_cell_vertices ||
                    static_cast<std::size_t>(local_b) >= n_cell_vertices) {
                    continue;
                }
                bind_edge_interior(
                    cell,
                    local_edge,
                    cell_vertices[static_cast<std::size_t>(local_a)],
                    cell_vertices[static_cast<std::size_t>(local_b)]);
            }
        }

        const auto cell_geometry = mesh.cell_interior_geometry_dofs(cell);
        const auto cell_dofs =
            entity_map.getCellInteriorDofs(static_cast<GlobalIndex>(cell));
        if (!cell_dofs.empty()) {
            if (cell_dofs.size() != cell_geometry.size()) {
                throw std::invalid_argument(
                    "level-set signed-distance repair cell-interior DOF count does not match high-order mesh nodes");
            }
            for (std::size_t j = 0; j < cell_geometry.size(); ++j) {
                bind(cell_geometry[j], cell_dofs[j]);
            }
        }
    }

    for (svmp::index_t face = 0;
         face < static_cast<svmp::index_t>(mesh.n_faces());
         ++face) {
        const auto face_dofs =
            entity_map.getFaceDofs(static_cast<GlobalIndex>(face));
        if (face_dofs.empty()) {
            continue;
        }
        const auto face_geometry = faceInteriorGeometryNodes(mesh, face);
        if (face_dofs.size() != face_geometry.size()) {
            throw std::invalid_argument(
                "level-set signed-distance repair face DOF count does not match high-order mesh face nodes");
        }
        for (std::size_t j = 0; j < face_geometry.size(); ++j) {
            bind(face_geometry[j], face_dofs[j]);
        }
    }
}
#endif

/// Bind every field DOF to a coordinate through the isoparametric nodal
/// pairing getCellDofs(cell)[i] <-> i-th cell node. Returns false without
/// invoking the callback when any cell breaks the pairing (non-nodal or
/// sub/super-parametric fields), so callers can fall back to entity-aware
/// binding.
template <typename Callback>
[[nodiscard]] bool tryForEachCellNodalDofPoint(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& field_dofs,
    std::size_t coefficient_count,
    const Callback& repair_dof_at_point)
{
    bool pairable = true;
    std::vector<GlobalIndex> cell_nodes;
    std::vector<std::array<Real, 3>> cell_coordinates;
    mesh.forEachCell([&](GlobalIndex cell) {
        if (!pairable) {
            return;
        }
        mesh.getCellNodes(cell, cell_nodes);
        mesh.getCellCoordinates(cell, cell_coordinates);
        const auto cell_dofs = field_dofs.getCellDofs(cell);
        if (cell_nodes.empty() ||
            cell_dofs.size() != cell_nodes.size() ||
            cell_coordinates.size() < cell_nodes.size()) {
            pairable = false;
            return;
        }
        for (const auto dof : cell_dofs) {
            if (dof < 0 ||
                static_cast<std::size_t>(dof) >= coefficient_count) {
                pairable = false;
                return;
            }
        }
    });
    if (!pairable) {
        return false;
    }

    mesh.forEachCell([&](GlobalIndex cell) {
        mesh.getCellNodes(cell, cell_nodes);
        mesh.getCellCoordinates(cell, cell_coordinates);
        const auto cell_dofs = field_dofs.getCellDofs(cell);
        for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
            repair_dof_at_point(cell_dofs[i], cell_coordinates[i]);
        }
    });
    return true;
}

} // namespace

LevelSetSignedDistanceRepairResult repairLevelSetSignedDistanceByProjection(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetReinitializationOptions& options,
    std::span<const Real> input_coefficients,
    std::vector<Real>& repaired_coefficients)
{
    rejectDistributedProjectionReinitialization(level_set_dofs);

    const auto expected = static_cast<std::size_t>(level_set_dofs.getNumDofs());
    if (!(options.signed_distance_tolerance > 0.0) ||
        !std::isfinite(options.signed_distance_tolerance)) {
        throw std::invalid_argument(
            "level-set signed-distance repair requires a positive signed-distance tolerance");
    }
    if (options.max_iterations <= 0) {
        throw std::invalid_argument(
            "level-set signed-distance repair requires positive max_iterations");
    }
    if (!(options.pseudo_time_step_scale > 0.0) ||
        !(options.pseudo_time_step_scale <= 1.0) ||
        !std::isfinite(options.pseudo_time_step_scale)) {
        throw std::invalid_argument(
            "level-set signed-distance repair requires pseudo_time_step_scale in (0, 1]");
    }
    if (options.preserve_band_width > 0.0) {
        throw std::invalid_argument(
            "level-set signed-distance repair no longer supports preserve_band_width; "
            "use max_zero_set_displacement to bound the redistance update instead");
    }
    if (!(options.interface_band_width > 0.0) ||
        !std::isfinite(options.interface_band_width)) {
        throw std::invalid_argument(
            "level-set signed-distance repair requires a finite positive interface band width");
    }
    if (!(options.max_zero_set_displacement >= 0.0) ||
        !std::isfinite(options.max_zero_set_displacement)) {
        throw std::invalid_argument(
            "level-set signed-distance repair requires a finite nonnegative zero-set displacement bound");
    }
    if (input_coefficients.size() != expected) {
        throw std::invalid_argument(
            "level-set signed-distance repair received an incompatible coefficient span");
    }
    const auto* entity_map = level_set_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::invalid_argument(
            "level-set signed-distance repair requires a scalar nodal field");
    }

    repaired_coefficients.assign(input_coefficients.begin(), input_coefficients.end());

    const auto primitive_set =
        buildLinearInterfacePrimitives(mesh,
                                       *entity_map,
                                       options.signed_distance_tolerance,
                                       input_coefficients);
    return repairSignedDistanceCoefficientsFromPrimitives(
        mesh,
        level_set_dofs,
        *entity_map,
        options,
        input_coefficients,
        repaired_coefficients,
        primitive_set,
        [&](const auto& repair_dof_at_point) {
            for (GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
                const auto vertex_dofs =
                    scalarVertexDofSpan(*entity_map, vertex, expected);
                if (vertex_dofs.empty()) {
                    continue;
                }
                repair_dof_at_point(vertex_dofs.front(),
                                    mesh.getNodeCoordinates(vertex));
            }
        });
}

LevelSetSignedDistanceRepairResult repairLevelSetSignedDistanceByProjection(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetReinitializationOptions& options,
    std::span<const Real> input_solution,
    std::vector<Real>& repaired_solution)
{
    rejectDistributedProjectionReinitialization(system.dofHandler());

    if (!(options.signed_distance_tolerance > 0.0) ||
        !std::isfinite(options.signed_distance_tolerance) ||
        options.max_iterations <= 0 ||
        !(options.pseudo_time_step_scale > 0.0) ||
        !(options.pseudo_time_step_scale <= 1.0) ||
        !std::isfinite(options.pseudo_time_step_scale) ||
        !(options.interface_band_width > 0.0) ||
        !std::isfinite(options.interface_band_width) ||
        options.preserve_band_width > 0.0 ||
        !(options.max_zero_set_displacement >= 0.0) ||
        !std::isfinite(options.max_zero_set_displacement)) {
        throw std::invalid_argument(
            "level-set signed-distance repair received invalid tolerance, iteration, relaxation, preservation, or zero-set bound options");
    }
    const auto& field_dofs = system.fieldDofHandler(level_set_field);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(level_set_field));
    if (offset + n_field_dofs > input_solution.size()) {
        throw std::invalid_argument(
            "level-set signed-distance repair received an incompatible system solution span");
    }

    std::vector<Real> field_coefficients(n_field_dofs, 0.0);
    std::copy_n(input_solution.begin() + static_cast<std::ptrdiff_t>(offset),
                n_field_dofs,
                field_coefficients.begin());

    std::vector<Real> repaired_field;
    repaired_field.assign(field_coefficients.begin(), field_coefficients.end());
    const auto& mesh_access = system.meshAccess();
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::invalid_argument(
            "level-set signed-distance repair requires a scalar nodal field");
    }
    const auto primitive_set =
        buildLinearInterfacePrimitives(mesh_access,
                                       *entity_map,
                                       options.signed_distance_tolerance,
                                       field_coefficients);

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        auto result = repairSignedDistanceCoefficientsFromPrimitives(
            mesh_access,
            field_dofs,
            *entity_map,
            options,
            std::span<const Real>(field_coefficients.data(),
                                  field_coefficients.size()),
            repaired_field,
            primitive_set,
            [&](const auto& repair_dof_at_point) {
                for (GlobalIndex vertex = 0;
                     vertex < mesh_access.numVertices();
                     ++vertex) {
                    const auto vertex_dofs =
                        scalarVertexDofSpan(*entity_map,
                                            vertex,
                                            field_coefficients.size());
                    if (vertex_dofs.empty()) {
                        continue;
                    }
                    repair_dof_at_point(
                        vertex_dofs.front(),
                        mesh_access.getNodeCoordinates(vertex));
                }
            });
        repaired_solution.assign(input_solution.begin(), input_solution.end());
        std::copy(repaired_field.begin(),
                  repaired_field.end(),
                  repaired_solution.begin() + static_cast<std::ptrdiff_t>(offset));
        return result;
    }

    auto result = repairSignedDistanceCoefficientsFromPrimitives(
        mesh_access,
        field_dofs,
        *entity_map,
        options,
        std::span<const Real>(field_coefficients.data(),
                              field_coefficients.size()),
        repaired_field,
        primitive_set,
        [&](const auto& repair_dof_at_point) {
            // Isoparametric nodal Lagrange fields pair getCellDofs(cell)[i]
            // with the cell's i-th mesh node — the same convention used to
            // build level-set fields and consumed by the cut backends. Prefer
            // it over the entity-id walk below: entity-aware binding assumes
            // the FE edge/face numbering matches the mesh tables, which does
            // not hold on all native meshes and then pairs DOFs with the
            // wrong coordinates. The walk remains the fallback for
            // sub/super-parametric fields where DOF and node counts differ.
            if (tryForEachCellNodalDofPoint(mesh_access,
                                            field_dofs,
                                            field_coefficients.size(),
                                            repair_dof_at_point)) {
                return;
            }
            forEachNativeMeshScalarDofPoint(
                native_mesh->local_mesh(),
                mesh_access,
                *entity_map,
                field_coefficients.size(),
                repair_dof_at_point);
        });
#else
    auto result = repairSignedDistanceCoefficientsFromPrimitives(
        mesh_access,
        field_dofs,
        *entity_map,
        options,
        std::span<const Real>(field_coefficients.data(),
                              field_coefficients.size()),
        repaired_field,
        primitive_set,
        [&](const auto& repair_dof_at_point) {
            for (GlobalIndex vertex = 0;
                 vertex < mesh_access.numVertices();
                 ++vertex) {
                const auto vertex_dofs =
                    scalarVertexDofSpan(*entity_map,
                                        vertex,
                                        field_coefficients.size());
                if (vertex_dofs.empty()) {
                    continue;
                }
                repair_dof_at_point(vertex_dofs.front(),
                                    mesh_access.getNodeCoordinates(vertex));
            }
        });
#endif

    repaired_solution.assign(input_solution.begin(), input_solution.end());
    std::copy(repaired_field.begin(),
              repaired_field.end(),
              repaired_solution.begin() + static_cast<std::ptrdiff_t>(offset));
    return result;
}

} // namespace svmp::FE::level_set
