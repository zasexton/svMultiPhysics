#include "LevelSet/LevelSetReinitialization.h"

#include "Dofs/EntityDofMap.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace svmp::FE::level_set {
namespace {

using interfaces::CutInterfaceDomainRequest;
using interfaces::CutInterfaceFragmentKind;
using interfaces::LevelSetCellCutInput;
using interfaces::LevelSetInterfaceSource;

struct SurfacePrimitive {
    CutInterfaceFragmentKind kind{CutInterfaceFragmentKind::Segment};
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

void setCoefficientAtVertex(const dofs::EntityDofMap& entity_map,
                            GlobalIndex vertex,
                            Real value,
                            std::vector<Real>& coefficients)
{
    const auto dofs = entity_map.getVertexDofs(vertex);
    const auto dof = dofs.front();
    coefficients[static_cast<std::size_t>(dof)] = value;
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

} // namespace

LevelSetSignedDistanceRepairResult repairLevelSetSignedDistanceByProjection(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetReinitializationOptions& options,
    std::span<const Real> input_coefficients,
    std::vector<Real>& repaired_coefficients)
{
    const auto expected = static_cast<std::size_t>(level_set_dofs.getNumDofs());
    if (!(options.signed_distance_tolerance > 0.0)) {
        throw std::invalid_argument(
            "level-set signed-distance repair requires a positive signed-distance tolerance");
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
    if (entity_map->numVertices() != mesh.numVertices()) {
        throw std::invalid_argument(
            "level-set signed-distance repair requires field and mesh vertex counts to match");
    }

    repaired_coefficients.assign(input_coefficients.begin(), input_coefficients.end());

    CutInterfaceDomainRequest request{};
    request.source = LevelSetInterfaceSource::fromField(FieldId{0});
    request.interface_marker = 0;
    request.tolerance = options.signed_distance_tolerance;
    request.quadrature_order = 1;

    std::vector<SurfacePrimitive> primitives;
    std::size_t cut_cells = 0u;
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
                coefficientAtVertex(*entity_map,
                                    cell_nodes[i],
                                    input_coefficients));
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
            primitive.points.reserve(fragment.vertices.size());
            for (const auto& vertex : fragment.vertices) {
                primitive.points.push_back(vertex.point);
            }
            primitives.push_back(std::move(primitive));
            added_cell_fragment = true;
        }
        if (added_cell_fragment) {
            ++cut_cells;
        }
    });

    LevelSetSignedDistanceRepairResult result;
    result.method = LevelSetReinitializationMethod::Projection;
    result.interface_fragments = primitives.size();
    result.cut_cells = cut_cells;
    if (primitives.empty()) {
        result.success = false;
        result.diagnostic = "level-set signed-distance repair found no active interface fragments";
        return result;
    }

    Real interface_displacement_squared_sum = 0.0;
    for (GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto original = coefficientAtVertex(*entity_map, vertex, input_coefficients);
        const auto x = mesh.getNodeCoordinates(vertex);
        const Real d = nearestDistanceToInterface(x, primitives);
        if (!std::isfinite(d)) {
            throw std::runtime_error(
                "level-set signed-distance repair produced a non-finite distance");
        }
        Real repaired = 0.0;
        if (original > options.signed_distance_tolerance) {
            repaired = d;
        } else if (original < -options.signed_distance_tolerance) {
            repaired = -d;
        }
        setCoefficientAtVertex(*entity_map, vertex, repaired, repaired_coefficients);
        const Real abs_update = std::abs(repaired - original);
        result.max_abs_update = std::max(result.max_abs_update, abs_update);
        result.max_distance = std::max(result.max_distance, d);
        if (options.interface_band_width > Real{0.0} &&
            std::abs(original) <= options.interface_band_width) {
            result.max_interface_displacement =
                std::max(result.max_interface_displacement, abs_update);
            interface_displacement_squared_sum += abs_update * abs_update;
            ++result.interface_displacement_samples;
        }
        ++result.repaired_dofs;
    }
    if (result.interface_displacement_samples > 0u) {
        result.l2_interface_displacement =
            std::sqrt(interface_displacement_squared_sum /
                      static_cast<Real>(result.interface_displacement_samples));
    }
    result.success = true;
    return result;
}

LevelSetSignedDistanceRepairResult repairLevelSetSignedDistanceByProjection(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetReinitializationOptions& options,
    std::span<const Real> input_solution,
    std::vector<Real>& repaired_solution)
{
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
    auto result = repairLevelSetSignedDistanceByProjection(system.meshAccess(),
                                                           field_dofs,
                                                           options,
                                                           field_coefficients,
                                                           repaired_field);

    repaired_solution.assign(input_solution.begin(), input_solution.end());
    std::copy(repaired_field.begin(),
              repaired_field.end(),
              repaired_solution.begin() + static_cast<std::ptrdiff_t>(offset));
    return result;
}

} // namespace svmp::FE::level_set
