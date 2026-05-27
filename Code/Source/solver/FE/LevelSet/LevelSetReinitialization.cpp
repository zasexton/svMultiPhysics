#include "LevelSet/LevelSetReinitialization.h"

#include "Dofs/EntityDofMap.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
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

struct LinearInterfacePrimitiveSet {
    std::vector<SurfacePrimitive> primitives{};
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
            primitive.points.reserve(fragment.vertices.size());
            for (const auto& vertex : fragment.vertices) {
                primitive.points.push_back(vertex.point);
            }
            output.primitives.push_back(std::move(primitive));
            added_cell_fragment = true;
        }
        if (added_cell_fragment) {
            ++output.cut_cells;
        }
    });
    return output;
}

template <typename ForEachDofPoint>
[[nodiscard]] LevelSetSignedDistanceRepairResult repairSignedDistanceCoefficientsFromPrimitives(
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
    Real interface_displacement_squared_sum = 0.0;
    std::vector<unsigned char> repaired_once(expected, 0u);

    const auto repair_dof_at_point = [&](GlobalIndex dof,
                                         const std::array<Real, 3>& x) {
        if (dof < 0 || static_cast<std::size_t>(dof) >= expected) {
            throw std::invalid_argument(
                "level-set signed-distance repair found a cell DOF outside the coefficient span");
        }
        const auto dof_index = static_cast<std::size_t>(dof);
        if (repaired_once[dof_index] != 0u) {
            return;
        }

        const auto original = input_coefficients[dof_index];
        const Real d = nearestDistanceToInterface(x, primitive_set.primitives);
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
        repaired_coefficients[dof_index] = repaired;
        repaired_once[dof_index] = 1u;

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
    };

    for_each_dof_point(repair_dof_at_point);

    const auto unrepaired =
        static_cast<std::size_t>(std::count(repaired_once.begin(),
                                            repaired_once.end(),
                                            static_cast<unsigned char>(0u)));
    if (unrepaired != 0u) {
        result.success = false;
        result.diagnostic =
            "level-set signed-distance repair left " +
            std::to_string(unrepaired) +
            " coefficient(s) without an entity-aware mesh-node binding";
        return result;
    }

    if (result.interface_displacement_samples > 0u) {
        result.l2_interface_displacement =
            std::sqrt(interface_displacement_squared_sum /
                      static_cast<Real>(result.interface_displacement_samples));
    }
    result.success = true;
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
    const auto bind = [&](svmp::index_t geometry_node, GlobalIndex dof) {
        if (geometry_node < 0 ||
            static_cast<std::size_t>(geometry_node) >=
                static_cast<std::size_t>(mesh.n_vertices())) {
            throw std::invalid_argument(
                "level-set signed-distance repair found a mesh geometry node outside the mesh");
        }
        if (dof < 0 || static_cast<std::size_t>(dof) >= coefficient_count) {
            throw std::invalid_argument(
                "level-set signed-distance repair found an entity DOF outside the coefficient span");
        }
        const auto sdof = static_cast<std::size_t>(dof);
        if (bound[sdof] != 0u) {
            return;
        }
        callback(dof, coordinate_access.getNodeCoordinates(geometry_node));
        bound[sdof] = 1u;
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
        if (edge_geometry.size() <= 2u) {
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
        const auto edge_dofs = entity_map.getEdgeDofs(edge);
        if (edge_dofs.empty()) {
            return;
        }
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

    repaired_coefficients.assign(input_coefficients.begin(), input_coefficients.end());

    const auto primitive_set =
        buildLinearInterfacePrimitives(mesh,
                                       *entity_map,
                                       options.signed_distance_tolerance,
                                       input_coefficients);
    return repairSignedDistanceCoefficientsFromPrimitives(
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
        options,
        std::span<const Real>(field_coefficients.data(),
                              field_coefficients.size()),
        repaired_field,
        primitive_set,
        [&](const auto& repair_dof_at_point) {
            forEachNativeMeshScalarDofPoint(
                native_mesh->local_mesh(),
                mesh_access,
                *entity_map,
                field_coefficients.size(),
                repair_dof_at_point);
        });
#else
    auto result = repairSignedDistanceCoefficientsFromPrimitives(
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
