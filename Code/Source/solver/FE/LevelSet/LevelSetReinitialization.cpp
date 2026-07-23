#include "LevelSet/LevelSetReinitialization.h"

#include "Dofs/EntityDofMap.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
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

#if FE_HAS_MPI
[[nodiscard]] bool usesMultipleRanks(const dofs::DofHandler& dof_handler)
{
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (initialized == 0 || finalized != 0 ||
        dof_handler.mpiComm() == MPI_COMM_NULL) {
        return false;
    }

    int communicator_size = 1;
    MPI_Comm_size(dof_handler.mpiComm(), &communicator_size);
    return communicator_size > 1;
}

[[nodiscard]] MPI_Datatype mpiRealType()
{
    if constexpr (std::is_same_v<Real, double>) {
        return MPI_DOUBLE;
    } else if constexpr (std::is_same_v<Real, float>) {
        return MPI_FLOAT;
    } else {
        return MPI_LONG_DOUBLE;
    }
}
#endif

struct SurfacePrimitive {
    CutInterfaceFragmentKind kind{CutInterfaceFragmentKind::Segment};
    GlobalIndex parent_cell{-1};
    std::vector<std::array<Real, 3>> points{};
};

struct EdgeZeroCrossing {
    GlobalIndex dof_a{-1};
    GlobalIndex dof_b{-1};
    std::array<Real, 3> point_a{};
    std::array<Real, 3> point_b{};
    Real original_t{0.0};
};

struct CutCellPrimitiveData {
    int owner_rank{0};
    GlobalIndex parent_cell{-1};
    ElementType element_type{ElementType::Unknown};
    std::vector<GlobalIndex> dofs{};
    std::vector<SurfacePrimitive> primitives{};
    std::vector<EdgeZeroCrossing> crossings{};
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
    std::vector<CutCellPrimitiveData> cut_cells{};
};

[[nodiscard]] std::vector<std::array<std::size_t, 2>> cornerEdges(
    ElementType type);

[[nodiscard]] LinearInterfacePrimitiveSet buildLinearInterfacePrimitives(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& field_dofs,
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
    mesh.forEachOwnedCell([&](GlobalIndex cell_id) {
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

        CutCellPrimitiveData cell_data;
        cell_data.owner_rank = mesh.parallelRank();
        cell_data.parent_cell = mesh.globalEntityIdsAvailable()
                                    ? mesh.getCellGlobalId(cell_id)
                                    : cell_id;
        cell_data.element_type = type;
        const auto cell_dofs = field_dofs.getCellDofs(cell_id);
        cell_data.dofs.assign(cell_dofs.begin(), cell_dofs.end());
        for (const auto& fragment : cut_result.fragments) {
            if (!fragment.active()) {
                continue;
            }
            SurfacePrimitive primitive;
            primitive.kind = fragment.kind;
            primitive.parent_cell = cell_data.parent_cell;
            primitive.points.reserve(fragment.vertices.size());
            for (const auto& vertex : fragment.vertices) {
                primitive.points.push_back(vertex.point);
            }
            output.primitives.push_back(primitive);
            cell_data.primitives.push_back(std::move(primitive));
        }
        if (cell_data.primitives.empty()) {
            return;
        }

        for (const auto edge : cornerEdges(type)) {
            if (edge[0] >= count || edge[1] >= count) {
                continue;
            }
            const auto dofs_a = entity_map.getVertexDofs(cell_nodes[edge[0]]);
            const auto dofs_b = entity_map.getVertexDofs(cell_nodes[edge[1]]);
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
            const Real denominator = a - b;
            if (std::abs(denominator) <= tolerance) {
                continue;
            }
            EdgeZeroCrossing crossing{
                .dof_a = dofs_a.front(),
                .dof_b = dofs_b.front(),
                .point_a = cell_coordinates[edge[0]],
                .point_b = cell_coordinates[edge[1]],
                .original_t = std::clamp(a / denominator,
                                         Real{0.0},
                                         Real{1.0}),
            };
            if (crossing.dof_b < crossing.dof_a) {
                std::swap(crossing.dof_a, crossing.dof_b);
                std::swap(crossing.point_a, crossing.point_b);
                crossing.original_t = Real{1.0} - crossing.original_t;
            }
            cell_data.crossings.push_back(std::move(crossing));
        }
        output.cut_cells.push_back(std::move(cell_data));
    });
    return output;
}

#if FE_HAS_MPI
template <typename T>
void appendPod(std::vector<std::byte>& bytes, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    const auto old_size = bytes.size();
    bytes.resize(old_size + sizeof(T));
    std::memcpy(bytes.data() + old_size, &value, sizeof(T));
}

template <typename T>
[[nodiscard]] T readPod(std::span<const std::byte> bytes,
                        std::size_t& offset)
{
    static_assert(std::is_trivially_copyable_v<T>);
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
        throw std::runtime_error(
            "distributed level-set reinitialization received a truncated primitive snapshot");
    }
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

[[nodiscard]] std::vector<std::byte> serializePrimitiveSet(
    const LinearInterfacePrimitiveSet& set)
{
    std::vector<std::byte> bytes;
    appendPod(bytes, static_cast<std::uint64_t>(set.cut_cells.size()));
    for (const auto& cell : set.cut_cells) {
        appendPod(bytes, static_cast<std::int32_t>(cell.owner_rank));
        appendPod(bytes, static_cast<std::int64_t>(cell.parent_cell));
        appendPod(bytes, static_cast<std::uint8_t>(cell.element_type));
        appendPod(bytes, static_cast<std::uint64_t>(cell.dofs.size()));
        for (const auto dof : cell.dofs) {
            appendPod(bytes, static_cast<std::int64_t>(dof));
        }
        appendPod(bytes, static_cast<std::uint64_t>(cell.primitives.size()));
        for (const auto& primitive : cell.primitives) {
            appendPod(bytes, static_cast<std::uint8_t>(primitive.kind));
            appendPod(bytes, static_cast<std::int64_t>(primitive.parent_cell));
            appendPod(bytes,
                      static_cast<std::uint64_t>(primitive.points.size()));
            for (const auto& point : primitive.points) {
                appendPod(bytes, point[0]);
                appendPod(bytes, point[1]);
                appendPod(bytes, point[2]);
            }
        }
        appendPod(bytes, static_cast<std::uint64_t>(cell.crossings.size()));
        for (const auto& crossing : cell.crossings) {
            appendPod(bytes, static_cast<std::int64_t>(crossing.dof_a));
            appendPod(bytes, static_cast<std::int64_t>(crossing.dof_b));
            for (const auto value : crossing.point_a) {
                appendPod(bytes, value);
            }
            for (const auto value : crossing.point_b) {
                appendPod(bytes, value);
            }
            appendPod(bytes, crossing.original_t);
        }
    }
    return bytes;
}

void appendDeserializedPrimitiveSet(std::span<const std::byte> bytes,
                                    LinearInterfacePrimitiveSet& output)
{
    std::size_t offset = 0u;
    const auto cell_count = readPod<std::uint64_t>(bytes, offset);
    for (std::uint64_t cell_index = 0; cell_index < cell_count; ++cell_index) {
        CutCellPrimitiveData cell;
        cell.owner_rank = readPod<std::int32_t>(bytes, offset);
        cell.parent_cell = static_cast<GlobalIndex>(
            readPod<std::int64_t>(bytes, offset));
        cell.element_type = static_cast<ElementType>(
            readPod<std::uint8_t>(bytes, offset));

        const auto dof_count = readPod<std::uint64_t>(bytes, offset);
        cell.dofs.reserve(static_cast<std::size_t>(dof_count));
        for (std::uint64_t i = 0; i < dof_count; ++i) {
            cell.dofs.push_back(static_cast<GlobalIndex>(
                readPod<std::int64_t>(bytes, offset)));
        }

        const auto primitive_count = readPod<std::uint64_t>(bytes, offset);
        cell.primitives.reserve(static_cast<std::size_t>(primitive_count));
        for (std::uint64_t i = 0; i < primitive_count; ++i) {
            SurfacePrimitive primitive;
            primitive.kind = static_cast<CutInterfaceFragmentKind>(
                readPod<std::uint8_t>(bytes, offset));
            primitive.parent_cell = static_cast<GlobalIndex>(
                readPod<std::int64_t>(bytes, offset));
            const auto point_count = readPod<std::uint64_t>(bytes, offset);
            primitive.points.reserve(static_cast<std::size_t>(point_count));
            for (std::uint64_t point = 0; point < point_count; ++point) {
                primitive.points.push_back({{
                    readPod<Real>(bytes, offset),
                    readPod<Real>(bytes, offset),
                    readPod<Real>(bytes, offset),
                }});
            }
            cell.primitives.push_back(std::move(primitive));
        }

        const auto crossing_count = readPod<std::uint64_t>(bytes, offset);
        cell.crossings.reserve(static_cast<std::size_t>(crossing_count));
        for (std::uint64_t i = 0; i < crossing_count; ++i) {
            EdgeZeroCrossing crossing;
            crossing.dof_a = static_cast<GlobalIndex>(
                readPod<std::int64_t>(bytes, offset));
            crossing.dof_b = static_cast<GlobalIndex>(
                readPod<std::int64_t>(bytes, offset));
            for (auto& value : crossing.point_a) {
                value = readPod<Real>(bytes, offset);
            }
            for (auto& value : crossing.point_b) {
                value = readPod<Real>(bytes, offset);
            }
            crossing.original_t = readPod<Real>(bytes, offset);
            cell.crossings.push_back(std::move(crossing));
        }
        output.cut_cells.push_back(std::move(cell));
    }
    if (offset != bytes.size()) {
        throw std::runtime_error(
            "distributed level-set reinitialization received trailing primitive snapshot bytes");
    }
}
#endif

[[nodiscard]] LinearInterfacePrimitiveSet globalizePrimitiveSet(
    const dofs::DofHandler& field_dofs,
    LinearInterfacePrimitiveSet local)
{
#if FE_HAS_MPI
    if (usesMultipleRanks(field_dofs)) {
        const auto local_bytes = serializePrimitiveSet(local);
        if (local_bytes.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "distributed level-set reinitialization primitive snapshot exceeds MPI count capacity");
        }

        int communicator_size = 1;
        MPI_Comm_size(field_dofs.mpiComm(), &communicator_size);
        const int local_count = static_cast<int>(local_bytes.size());
        std::vector<int> counts(static_cast<std::size_t>(communicator_size), 0);
        MPI_Allgather(&local_count,
                      1,
                      MPI_INT,
                      counts.data(),
                      1,
                      MPI_INT,
                      field_dofs.mpiComm());
        std::vector<int> displacements(counts.size(), 0);
        std::size_t total = 0u;
        for (std::size_t rank = 0; rank < counts.size(); ++rank) {
            if (counts[rank] < 0 ||
                total > static_cast<std::size_t>(
                            std::numeric_limits<int>::max() - counts[rank])) {
                throw std::runtime_error(
                    "distributed level-set reinitialization primitive gather exceeds MPI displacement capacity");
            }
            displacements[rank] = static_cast<int>(total);
            total += static_cast<std::size_t>(counts[rank]);
        }
        std::vector<std::byte> gathered(total);
        MPI_Allgatherv(local_bytes.data(),
                       local_count,
                       MPI_BYTE,
                       gathered.data(),
                       counts.data(),
                       displacements.data(),
                       MPI_BYTE,
                       field_dofs.mpiComm());

        LinearInterfacePrimitiveSet global;
        for (std::size_t rank = 0; rank < counts.size(); ++rank) {
            const auto begin = static_cast<std::size_t>(displacements[rank]);
            const auto count = static_cast<std::size_t>(counts[rank]);
            appendDeserializedPrimitiveSet(
                std::span<const std::byte>(gathered.data() + begin, count),
                global);
        }
        std::sort(global.cut_cells.begin(),
                  global.cut_cells.end(),
                  [](const auto& a, const auto& b) {
                      return std::pair{a.parent_cell, a.owner_rank} <
                             std::pair{b.parent_cell, b.owner_rank};
                  });
        for (const auto& cell : global.cut_cells) {
            global.primitives.insert(global.primitives.end(),
                                     cell.primitives.begin(),
                                     cell.primitives.end());
        }
        return global;
    }
#else
    (void)field_dofs;
#endif
    return local;
}

[[nodiscard]] auto wallContactConstraintKey(
    const LevelSetWallContactConstraint& constraint) noexcept
{
    return std::tuple{constraint.parent_cell_global_id,
                      constraint.interface_marker,
                      constraint.boundary_marker,
                      static_cast<std::uint8_t>(constraint.kind)};
}

[[nodiscard]] bool finiteVector(
    const std::array<Real, 3>& values) noexcept
{
    return std::all_of(values.begin(), values.end(), [](Real value) {
        return std::isfinite(value);
    });
}

[[nodiscard]] bool sameWallContactConstraintPayload(
    const LevelSetWallContactConstraint& left,
    const LevelSetWallContactConstraint& right) noexcept
{
    return left.geometry_revision == right.geometry_revision &&
           left.target_angle_radians == right.target_angle_radians &&
           left.physical_wall_normal == right.physical_wall_normal &&
           left.accepted_contact_point == right.accepted_contact_point &&
           left.accepted_contact_line_tangent ==
               right.accepted_contact_line_tangent;
}

void validateWallContactConstraint(
    const LevelSetWallContactConstraint& constraint)
{
    const bool known_kind =
        constraint.kind ==
            LevelSetWallContactConstraintKind::PrescribedAngle ||
        constraint.kind ==
            LevelSetWallContactConstraintKind::AcceptedDynamicAngle;
    if (!known_kind || constraint.interface_marker < 0 ||
        constraint.boundary_marker < 0 ||
        constraint.parent_cell_global_id == INVALID_GLOBAL_INDEX ||
        constraint.geometry_revision == 0u) {
        throw std::invalid_argument(
            "level-set wall-contact constraint requires a known kind, nonnegative markers, a valid global parent cell, and a nonzero geometry revision");
    }
    if (constraint.kind !=
        LevelSetWallContactConstraintKind::PrescribedAngle) {
        return;
    }

    const Real pi = std::acos(Real{-1.0});
    const Real wall_normal_norm = norm(constraint.physical_wall_normal);
    const Real line_tangent_norm =
        norm(constraint.accepted_contact_line_tangent);
    if (!std::isfinite(constraint.target_angle_radians) ||
        !(constraint.target_angle_radians > Real{0.0}) ||
        !(constraint.target_angle_radians < pi) ||
        !finiteVector(constraint.physical_wall_normal) ||
        !finiteVector(constraint.accepted_contact_point) ||
        !finiteVector(constraint.accepted_contact_line_tangent) ||
        !(wall_normal_norm > Real{0.0}) ||
        !std::isfinite(wall_normal_norm) ||
        !(line_tangent_norm > Real{0.0}) ||
        !std::isfinite(line_tangent_norm)) {
        throw std::invalid_argument(
            "prescribed level-set wall contact requires an angle in (0, pi), finite physical frame vectors, and nonzero wall-normal and contact-line directions");
    }
    const Real relative_alignment =
        std::abs(dot(constraint.physical_wall_normal,
                     constraint.accepted_contact_line_tangent)) /
        (wall_normal_norm * line_tangent_norm);
    if (relative_alignment > Real{1.0e-10}) {
        throw std::invalid_argument(
            "prescribed level-set wall contact requires an orthogonal physical wall normal and contact-line tangent");
    }
}

[[nodiscard]] std::vector<LevelSetWallContactConstraint>
globalizeWallContactConstraints(
    const dofs::DofHandler& field_dofs,
    std::span<const LevelSetWallContactConstraint> local_constraints)
{
    std::vector<LevelSetWallContactConstraint> constraints(
        local_constraints.begin(), local_constraints.end());
    for (const auto& constraint : constraints) {
        validateWallContactConstraint(constraint);
    }

#if FE_HAS_MPI
    if (usesMultipleRanks(field_dofs)) {
        std::vector<std::byte> local_bytes;
        appendPod(local_bytes,
                  static_cast<std::uint64_t>(constraints.size()));
        for (const auto& constraint : constraints) {
            appendPod(local_bytes,
                      static_cast<std::uint8_t>(constraint.kind));
            appendPod(local_bytes,
                      static_cast<std::int32_t>(constraint.interface_marker));
            appendPod(local_bytes,
                      static_cast<std::int32_t>(constraint.boundary_marker));
            appendPod(local_bytes,
                      static_cast<std::int64_t>(
                          constraint.parent_cell_global_id));
            appendPod(local_bytes, constraint.geometry_revision);
            appendPod(local_bytes, constraint.target_angle_radians);
            for (const Real value : constraint.physical_wall_normal) {
                appendPod(local_bytes, value);
            }
            for (const Real value : constraint.accepted_contact_point) {
                appendPod(local_bytes, value);
            }
            for (const Real value :
                 constraint.accepted_contact_line_tangent) {
                appendPod(local_bytes, value);
            }
        }
        if (local_bytes.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "distributed level-set wall-contact constraint snapshot exceeds MPI count capacity");
        }

        int communicator_size = 1;
        MPI_Comm_size(field_dofs.mpiComm(), &communicator_size);
        const int local_count = static_cast<int>(local_bytes.size());
        std::vector<int> counts(static_cast<std::size_t>(communicator_size),
                                0);
        MPI_Allgather(&local_count,
                      1,
                      MPI_INT,
                      counts.data(),
                      1,
                      MPI_INT,
                      field_dofs.mpiComm());
        std::vector<int> displacements(counts.size(), 0);
        std::size_t total = 0u;
        for (std::size_t rank = 0; rank < counts.size(); ++rank) {
            if (counts[rank] < 0 ||
                total > static_cast<std::size_t>(
                            std::numeric_limits<int>::max() - counts[rank])) {
                throw std::runtime_error(
                    "distributed level-set wall-contact constraint gather exceeds MPI displacement capacity");
            }
            displacements[rank] = static_cast<int>(total);
            total += static_cast<std::size_t>(counts[rank]);
        }
        std::vector<std::byte> gathered(total);
        MPI_Allgatherv(local_bytes.data(),
                       local_count,
                       MPI_BYTE,
                       gathered.data(),
                       counts.data(),
                       displacements.data(),
                       MPI_BYTE,
                       field_dofs.mpiComm());

        constraints.clear();
        for (std::size_t rank = 0; rank < counts.size(); ++rank) {
            const auto begin = static_cast<std::size_t>(displacements[rank]);
            const auto count = static_cast<std::size_t>(counts[rank]);
            const std::span<const std::byte> bytes(
                gathered.data() + begin, count);
            std::size_t offset = 0u;
            const auto constraint_count =
                readPod<std::uint64_t>(bytes, offset);
            for (std::uint64_t index = 0u; index < constraint_count;
                 ++index) {
                LevelSetWallContactConstraint constraint;
                constraint.kind =
                    static_cast<LevelSetWallContactConstraintKind>(
                        readPod<std::uint8_t>(bytes, offset));
                constraint.interface_marker =
                    readPod<std::int32_t>(bytes, offset);
                constraint.boundary_marker =
                    readPod<std::int32_t>(bytes, offset);
                constraint.parent_cell_global_id =
                    static_cast<GlobalIndex>(
                        readPod<std::int64_t>(bytes, offset));
                constraint.geometry_revision =
                    readPod<std::uint64_t>(bytes, offset);
                constraint.target_angle_radians =
                    readPod<Real>(bytes, offset);
                for (Real& value : constraint.physical_wall_normal) {
                    value = readPod<Real>(bytes, offset);
                }
                for (Real& value : constraint.accepted_contact_point) {
                    value = readPod<Real>(bytes, offset);
                }
                for (Real& value :
                     constraint.accepted_contact_line_tangent) {
                    value = readPod<Real>(bytes, offset);
                }
                validateWallContactConstraint(constraint);
                constraints.push_back(constraint);
            }
            if (offset != bytes.size()) {
                throw std::runtime_error(
                    "distributed level-set wall-contact constraint snapshot has trailing bytes");
            }
        }
    }
#else
    (void)field_dofs;
#endif

    std::sort(constraints.begin(),
              constraints.end(),
              [](const auto& left, const auto& right) {
                  return wallContactConstraintKey(left) <
                         wallContactConstraintKey(right);
              });
    std::vector<LevelSetWallContactConstraint> unique;
    unique.reserve(constraints.size());
    std::map<int, std::uint64_t> geometry_revision_by_interface;
    std::map<std::tuple<GlobalIndex, int, int>,
             LevelSetWallContactConstraintKind>
        kind_by_parent_and_wall;
    for (const auto& constraint : constraints) {
        const auto [revision, inserted_revision] =
            geometry_revision_by_interface.emplace(
                constraint.interface_marker,
                constraint.geometry_revision);
        if (!inserted_revision &&
            revision->second != constraint.geometry_revision) {
            throw std::invalid_argument(
                "distributed level-set wall-contact constraints mix geometry revisions for one interface");
        }
        const auto parent_wall_key = std::tuple{
            constraint.parent_cell_global_id,
            constraint.interface_marker,
            constraint.boundary_marker};
        const auto [kind, inserted_kind] =
            kind_by_parent_and_wall.emplace(parent_wall_key,
                                            constraint.kind);
        if (!inserted_kind && kind->second != constraint.kind) {
            throw std::invalid_argument(
                "distributed level-set wall-contact constraint assigns multiple contact laws to one parent wall");
        }
        if (!unique.empty() &&
            wallContactConstraintKey(unique.back()) ==
                wallContactConstraintKey(constraint)) {
            if (!sameWallContactConstraintPayload(unique.back(),
                                                  constraint)) {
                throw std::invalid_argument(
                    "distributed level-set wall-contact constraint has conflicting geometry or prescribed-frame data");
            }
            continue;
        }
        unique.push_back(constraint);
    }
    return unique;
}

[[nodiscard]] std::vector<Real> ownerSynchronizedCoefficients(
    const dofs::DofHandler& field_dofs,
    std::span<const Real> coefficients)
{
    std::vector<Real> synchronized(coefficients.begin(), coefficients.end());
#if FE_HAS_MPI
    if (!usesMultipleRanks(field_dofs)) {
        return synchronized;
    }
    if (synchronized.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "distributed level-set reinitialization coefficient vector exceeds MPI count capacity");
    }

    const auto& owned = field_dofs.getPartition().locallyOwned();
    std::vector<Real> local(synchronized.size(), Real{0.0});
    std::vector<int> local_owner_count(synchronized.size(), 0);
    for (std::size_t i = 0; i < synchronized.size(); ++i) {
        if (owned.contains(static_cast<GlobalIndex>(i))) {
            local[i] = synchronized[i];
            local_owner_count[i] = 1;
        }
    }
    std::vector<int> owner_count(synchronized.size(), 0);
    MPI_Allreduce(local.data(),
                  synchronized.data(),
                  static_cast<int>(synchronized.size()),
                  mpiRealType(),
                  MPI_SUM,
                  field_dofs.mpiComm());
    MPI_Allreduce(local_owner_count.data(),
                  owner_count.data(),
                  static_cast<int>(owner_count.size()),
                  MPI_INT,
                  MPI_SUM,
                  field_dofs.mpiComm());
    if (std::any_of(owner_count.begin(), owner_count.end(),
                    [](int count) { return count != 1; })) {
        throw std::runtime_error(
            "distributed level-set reinitialization requires exactly one owner for every coefficient");
    }
#else
    (void)field_dofs;
#endif
    return synchronized;
}

void synchronizeDofPoints(const dofs::DofHandler& field_dofs,
                          std::vector<unsigned char>& bound,
                          std::vector<std::array<Real, 3>>& points)
{
#if FE_HAS_MPI
    if (!usesMultipleRanks(field_dofs)) {
        return;
    }
    if (points.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max() / 3)) {
        throw std::runtime_error(
            "distributed level-set reinitialization point vector exceeds MPI count capacity");
    }

    const auto& owned = field_dofs.getPartition().locallyOwned();
    std::vector<Real> local(points.size() * 3u, Real{0.0});
    std::vector<Real> global(local.size(), Real{0.0});
    std::vector<int> local_owner_count(points.size(), 0);
    std::vector<int> owner_count(points.size(), 0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (bound[i] == 0u ||
            !owned.contains(static_cast<GlobalIndex>(i))) {
            continue;
        }
        local[3u * i] = points[i][0];
        local[3u * i + 1u] = points[i][1];
        local[3u * i + 2u] = points[i][2];
        local_owner_count[i] = 1;
    }
    MPI_Allreduce(local.data(),
                  global.data(),
                  static_cast<int>(global.size()),
                  mpiRealType(),
                  MPI_SUM,
                  field_dofs.mpiComm());
    MPI_Allreduce(local_owner_count.data(),
                  owner_count.data(),
                  static_cast<int>(owner_count.size()),
                  MPI_INT,
                  MPI_SUM,
                  field_dofs.mpiComm());
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (owner_count[i] != 1) {
            throw std::runtime_error(
                "distributed level-set reinitialization could not bind exactly one owner coordinate to every coefficient");
        }
        points[i] = {{global[3u * i],
                      global[3u * i + 1u],
                      global[3u * i + 2u]}};
        bound[i] = 1u;
    }
#else
    (void)field_dofs;
    (void)bound;
    (void)points;
#endif
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

struct ZeroSetDisplacementEvaluation {
    bool topology_preserved{true};
    Real max_displacement{0.0};
    Real l2_displacement{0.0};
    std::size_t samples{0u};
};

[[nodiscard]] std::vector<EdgeZeroCrossing> collectOriginalZeroCrossings(
    const LinearInterfacePrimitiveSet& primitive_set)
{
    std::vector<EdgeZeroCrossing> crossings;
    for (const auto& cell : primitive_set.cut_cells) {
        crossings.insert(crossings.end(),
                         cell.crossings.begin(),
                         cell.crossings.end());
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

struct PrescribedContactFrame {
    std::array<Real, 3> wall_normal{};
    std::array<Real, 3> line_tangent{};
    std::array<Real, 3> wall_conormal{};
    std::array<Real, 3> target_normal{};
};

[[nodiscard]] PrescribedContactFrame prescribedContactFrame(
    const LevelSetWallContactConstraint& constraint,
    int dimension)
{
    if (dimension != 2 && dimension != 3) {
        throw std::invalid_argument(
            "prescribed level-set wall contact requires a two- or three-dimensional mesh");
    }
    validateWallContactConstraint(constraint);
    PrescribedContactFrame frame;
    frame.wall_normal = scale(
        constraint.physical_wall_normal,
        Real{1.0} / norm(constraint.physical_wall_normal));
    frame.line_tangent = scale(
        constraint.accepted_contact_line_tangent,
        Real{1.0} / norm(constraint.accepted_contact_line_tangent));
    frame.wall_conormal = cross(frame.line_tangent, frame.wall_normal);
    const Real conormal_norm = norm(frame.wall_conormal);
    if (!(conormal_norm > Real{0.0}) || !std::isfinite(conormal_norm)) {
        throw std::invalid_argument(
            "prescribed level-set wall contact has a degenerate contact-line frame");
    }
    frame.wall_conormal =
        scale(frame.wall_conormal, Real{1.0} / conormal_norm);
    frame.target_normal = add(
        scale(frame.wall_normal,
              -std::cos(constraint.target_angle_radians)),
        scale(frame.wall_conormal,
              std::sin(constraint.target_angle_radians)));
    const Real target_norm = norm(frame.target_normal);
    if (!(target_norm > Real{0.0}) || !std::isfinite(target_norm)) {
        throw std::invalid_argument(
            "prescribed level-set wall contact produced a degenerate target normal");
    }
    frame.target_normal =
        scale(frame.target_normal, Real{1.0} / target_norm);

    if (dimension == 2) {
        const Real frame_tolerance = Real{1.0e-10};
        if (std::abs(frame.wall_normal[2]) > frame_tolerance ||
            std::hypot(frame.line_tangent[0], frame.line_tangent[1]) >
                frame_tolerance ||
            std::abs(frame.target_normal[2]) > frame_tolerance) {
            throw std::invalid_argument(
                "prescribed two-dimensional wall contact requires an in-plane wall normal and an out-of-plane contact-line tangent");
        }
    }
    return frame;
}

struct AffineFieldFit {
    bool valid{false};
    Real intercept{0.0};
    std::array<Real, 3> gradient{{0.0, 0.0, 0.0}};
    Real max_value_residual{0.0};
};

[[nodiscard]] AffineFieldFit fitAffineFieldOnCell(
    const CutCellPrimitiveData& cell,
    std::span<const std::array<Real, 3>> dof_points,
    std::span<const Real> coefficients,
    int dimension)
{
    AffineFieldFit fit;
    const std::size_t parameter_count =
        static_cast<std::size_t>(dimension + 1);
    if ((dimension != 2 && dimension != 3) ||
        cell.dofs.size() < parameter_count) {
        return fit;
    }

    std::array<std::array<Real, 5>, 4> system{};
    Real matrix_scale = Real{1.0};
    for (const GlobalIndex dof : cell.dofs) {
        if (dof < 0 ||
            static_cast<std::size_t>(dof) >= dof_points.size() ||
            static_cast<std::size_t>(dof) >= coefficients.size()) {
            return fit;
        }
        const auto index = static_cast<std::size_t>(dof);
        std::array<Real, 4> row{{Real{1.0},
                                 dof_points[index][0],
                                 dof_points[index][1],
                                 dof_points[index][2]}};
        for (std::size_t i = 0; i < parameter_count; ++i) {
            for (std::size_t j = 0; j < parameter_count; ++j) {
                system[i][j] += row[i] * row[j];
                matrix_scale =
                    std::max(matrix_scale, std::abs(system[i][j]));
            }
            system[i][parameter_count] +=
                row[i] * coefficients[index];
        }
    }

    const Real pivot_tolerance =
        Real{4096.0} * std::numeric_limits<Real>::epsilon() *
        matrix_scale;
    for (std::size_t column = 0; column < parameter_count; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1u;
             row < parameter_count;
             ++row) {
            if (std::abs(system[row][column]) >
                std::abs(system[pivot][column])) {
                pivot = row;
            }
        }
        if (!(std::abs(system[pivot][column]) > pivot_tolerance)) {
            return fit;
        }
        if (pivot != column) {
            std::swap(system[pivot], system[column]);
        }
        const Real inverse_pivot = Real{1.0} / system[column][column];
        for (std::size_t entry = column;
             entry <= parameter_count;
             ++entry) {
            system[column][entry] *= inverse_pivot;
        }
        for (std::size_t row = 0; row < parameter_count; ++row) {
            if (row == column) {
                continue;
            }
            const Real multiplier = system[row][column];
            for (std::size_t entry = column;
                 entry <= parameter_count;
                 ++entry) {
                system[row][entry] -=
                    multiplier * system[column][entry];
            }
        }
    }

    fit.intercept = system[0][parameter_count];
    for (int component = 0; component < dimension; ++component) {
        fit.gradient[static_cast<std::size_t>(component)] =
            system[static_cast<std::size_t>(component + 1)]
                  [parameter_count];
    }
    for (const GlobalIndex dof : cell.dofs) {
        const auto index = static_cast<std::size_t>(dof);
        const Real fitted_value =
            fit.intercept + dot(fit.gradient, dof_points[index]);
        fit.max_value_residual = std::max(
            fit.max_value_residual,
            std::abs(fitted_value - coefficients[index]));
    }
    fit.valid = finiteVector(fit.gradient) &&
                std::isfinite(fit.intercept) &&
                std::isfinite(fit.max_value_residual);
    return fit;
}

template <typename ForEachDofPoint>
[[nodiscard]] LevelSetSignedDistanceRepairResult repairSignedDistanceCoefficientsFromPrimitives(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& field_dofs,
    const LevelSetReinitializationOptions& options,
    std::span<const Real> input_coefficients,
    std::vector<Real>& repaired_coefficients,
    const LinearInterfacePrimitiveSet& primitive_set,
    std::span<const LevelSetWallContactConstraint>
        wall_contact_constraints,
    ForEachDofPoint&& for_each_dof_point)
{
    LevelSetSignedDistanceRepairResult result;
    result.method = LevelSetReinitializationMethod::Projection;
    result.interface_fragments = primitive_set.primitives.size();
    result.cut_cells = primitive_set.cut_cells.size();
    result.wall_contact_constraints = wall_contact_constraints.size();
    result.wall_contact_constraints_satisfied =
        wall_contact_constraints.empty();
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
    synchronizeDofPoints(field_dofs, bound, dof_points);

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

    std::set<GlobalIndex> constrained_parent_cells;
    std::map<GlobalIndex, LevelSetWallContactConstraint>
        wall_constraint_by_parent;
    for (const auto& constraint : wall_contact_constraints) {
        constrained_parent_cells.insert(
            constraint.parent_cell_global_id);
        const auto insertion = wall_constraint_by_parent.emplace(
            constraint.parent_cell_global_id, constraint);
        if (!insertion.second) {
            throw std::invalid_argument(
                "level-set wall-contact projection supports one physical contact frame per parent cell");
        }
    }
    std::set<GlobalIndex> matched_constrained_parent_cells;
    for (const auto& cell : primitive_set.cut_cells) {
        if (constrained_parent_cells.contains(cell.parent_cell)) {
            matched_constrained_parent_cells.insert(cell.parent_cell);
        }
    }
    if (matched_constrained_parent_cells != constrained_parent_cells) {
        result.success = false;
        result.diagnostic =
            "level-set wall-contact constraint does not match the accepted cut-cell snapshot";
        return result;
    }
    result.wall_contact_cells =
        matched_constrained_parent_cells.size();

    // In cut cells, use the supporting line/plane of the local interface
    // fragment.  Extending that geometry past the clipped cell/wall endpoint
    // avoids the radial-distance artifact that otherwise rotates normals at a
    // contact point.  Away from cut cells, retain the true nearest finite-
    // primitive distance.
    std::vector<Real> cut_support_distance(
        expected, std::numeric_limits<Real>::infinity());
    for (const auto& cell : primitive_set.cut_cells) {
        for (const auto dof : cell.dofs) {
            if (dof < 0 || static_cast<std::size_t>(dof) >= expected) {
                throw std::invalid_argument(
                    "level-set signed-distance repair found a cut-cell DOF outside the coefficient span");
            }
            auto& local_distance =
                cut_support_distance[static_cast<std::size_t>(dof)];
            for (const auto& primitive : cell.primitives) {
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
    DisjointSet wall_contact_components(expected);
    std::vector<unsigned char> in_cut_patch(expected, 0u);
    std::vector<unsigned char> in_wall_contact_patch(expected, 0u);
    std::vector<unsigned char> in_prescribed_contact_patch(expected, 0u);
    std::vector<unsigned char> in_dynamic_contact_patch(expected, 0u);
    std::vector<unsigned char> prescribed_target_bound(expected, 0u);
    std::vector<Real> prescribed_contact_target(expected, Real{0.0});
    for (const auto& cell : primitive_set.cut_cells) {
        std::optional<std::size_t> first;
        std::optional<std::size_t> first_dynamic_contact;
        const auto constraint_position =
            wall_constraint_by_parent.find(cell.parent_cell);
        const bool wall_contact_cell =
            constraint_position != wall_constraint_by_parent.end();
        const bool prescribed_contact_cell =
            wall_contact_cell &&
            constraint_position->second.kind ==
                LevelSetWallContactConstraintKind::PrescribedAngle;
        const bool dynamic_contact_cell =
            wall_contact_cell &&
            constraint_position->second.kind ==
                LevelSetWallContactConstraintKind::AcceptedDynamicAngle;
        std::optional<PrescribedContactFrame> prescribed_frame;
        if (prescribed_contact_cell) {
            prescribed_frame = prescribedContactFrame(
                constraint_position->second, mesh.dimension());
        }
        for (const auto dof : cell.dofs) {
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
            if (wall_contact_cell) {
                in_wall_contact_patch[index] = 1u;
            }
            if (dynamic_contact_cell) {
                in_dynamic_contact_patch[index] = 1u;
                if (first_dynamic_contact.has_value()) {
                    wall_contact_components.unite(*first_dynamic_contact,
                                                  index);
                } else {
                    first_dynamic_contact = index;
                }
            }
            if (prescribed_contact_cell) {
                in_prescribed_contact_patch[index] = 1u;
                const Real prescribed_value = dot(
                    prescribed_frame->target_normal,
                    sub(dof_points[index],
                        constraint_position->second
                            .accepted_contact_point));
                if (prescribed_target_bound[index] != 0u) {
                    const Real compatibility_tolerance =
                        Real{4096.0} *
                        std::numeric_limits<Real>::epsilon() *
                        std::max({Real{1.0},
                                  std::abs(prescribed_value),
                                  std::abs(
                                      prescribed_contact_target[index])});
                    if (std::abs(prescribed_value -
                                 prescribed_contact_target[index]) >
                        compatibility_tolerance) {
                        throw std::invalid_argument(
                            "prescribed level-set wall-contact frames assign incompatible targets to a shared degree of freedom");
                    }
                } else {
                    prescribed_contact_target[index] = prescribed_value;
                    prescribed_target_bound[index] = 1u;
                }
            }
        }
    }
    for (std::size_t i = 0; i < expected; ++i) {
        if (in_prescribed_contact_patch[i] != 0u &&
            in_dynamic_contact_patch[i] != 0u) {
            throw std::invalid_argument(
                "level-set wall-contact projection cannot overlap prescribed and accepted-dynamic contact patches");
        }
    }
    result.wall_contact_dofs = static_cast<std::size_t>(std::count(
        in_wall_contact_patch.begin(),
        in_wall_contact_patch.end(),
        static_cast<unsigned char>(1u)));

    std::map<std::size_t, bool> component_requires_common_scale;
    for (std::size_t i = 0; i < expected; ++i) {
        if (in_cut_patch[i] != 0u) {
            component_requires_common_scale.emplace(components.find(i), false);
        }
    }
    for (const auto& cell : primitive_set.cut_cells) {
        const bool high_order =
            cell.dofs.size() > cornerCount(cell.element_type);
        if (!high_order) {
            continue;
        }
        for (const auto dof : cell.dofs) {
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

    std::map<std::size_t, std::pair<Real, Real>> wall_contact_scale_sums;
    for (std::size_t i = 0; i < expected; ++i) {
        if (in_dynamic_contact_patch[i] == 0u) {
            continue;
        }
        auto& sums = wall_contact_scale_sums[
            wall_contact_components.find(i)];
        sums.first += input_coefficients[i] * signed_distance_target[i];
        sums.second += input_coefficients[i] * input_coefficients[i];
    }
    std::map<std::size_t, Real> wall_contact_target_scale;
    for (const auto& [root, sums] : wall_contact_scale_sums) {
        Real fitted = Real{1.0};
        if (sums.second > options.signed_distance_tolerance *
                              options.signed_distance_tolerance) {
            fitted = sums.first / sums.second;
        }
        if (!std::isfinite(fitted) || fitted <= Real{0.0}) {
            fitted = Real{1.0};
        }
        wall_contact_target_scale[root] = fitted;
    }

    std::vector<Real> target(expected, 0.0);
    std::vector<unsigned char> update_enabled(expected, 1u);
    for (std::size_t i = 0; i < expected; ++i) {
        if (in_cut_patch[i] != 0u) {
            const auto root = components.find(i);
            if (in_prescribed_contact_patch[i] != 0u) {
                target[i] = prescribed_contact_target[i];
            } else if (component_requires_common_scale.at(root)) {
                target[i] = component_target_scale.at(root) *
                            input_coefficients[i];
            } else if (in_dynamic_contact_patch[i] != 0u) {
                target[i] = wall_contact_target_scale.at(
                                wall_contact_components.find(i)) *
                            input_coefficients[i];
            } else {
                target[i] = signed_distance_target[i];
            }
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

    const auto crossings = collectOriginalZeroCrossings(primitive_set);
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
        const Real signed_distance_error =
            std::abs(repaired_coefficients[i] - signed_distance_target[i]);
        result.max_signed_distance_error = std::max(
            result.max_signed_distance_error,
            signed_distance_error);
        if (in_wall_contact_patch[i] != 0u) {
            result.max_wall_constrained_signed_distance_error = std::max(
                result.max_wall_constrained_signed_distance_error,
                signed_distance_error);
        } else {
            result.max_unconstrained_signed_distance_error = std::max(
                result.max_unconstrained_signed_distance_error,
                signed_distance_error);
        }
        result.max_iteration_residual = std::max(
            result.max_iteration_residual,
            std::abs(repaired_coefficients[i] - target[i]));
        ++result.repaired_dofs;
    }

    if (!wall_contact_constraints.empty()) {
        std::map<std::size_t, std::pair<Real, Real>> repaired_scale_sums;
        Real constraint_scale = Real{1.0};
        for (std::size_t i = 0; i < expected; ++i) {
            if (in_dynamic_contact_patch[i] == 0u) {
                continue;
            }
            auto& sums = repaired_scale_sums[
                wall_contact_components.find(i)];
            sums.first += input_coefficients[i] * repaired_coefficients[i];
            sums.second += input_coefficients[i] * input_coefficients[i];
            constraint_scale = std::max(
                {constraint_scale,
                 std::abs(input_coefficients[i]),
                 std::abs(repaired_coefficients[i])});
        }
        std::map<std::size_t, Real> repaired_scales;
        bool positive_scales = true;
        for (const auto& [root, sums] : repaired_scale_sums) {
            if (!(sums.second > options.signed_distance_tolerance *
                                   options.signed_distance_tolerance)) {
                positive_scales = false;
                continue;
            }
            const Real fitted = sums.first / sums.second;
            if (!std::isfinite(fitted) || !(fitted > Real{0.0})) {
                positive_scales = false;
                continue;
            }
            repaired_scales[root] = fitted;
        }
        for (std::size_t i = 0; i < expected; ++i) {
            if (in_dynamic_contact_patch[i] == 0u) {
                continue;
            }
            const auto found = repaired_scales.find(
                wall_contact_components.find(i));
            if (found == repaired_scales.end()) {
                positive_scales = false;
                continue;
            }
            result.max_wall_contact_scale_residual = std::max(
                result.max_wall_contact_scale_residual,
                std::abs(repaired_coefficients[i] -
                         found->second * input_coefficients[i]));
        }
        const Real constraint_tolerance =
            Real{4096.0} * std::numeric_limits<Real>::epsilon() *
            constraint_scale;
        const bool dynamic_constraints_satisfied =
            positive_scales &&
            result.max_wall_contact_scale_residual <=
                constraint_tolerance;

        bool prescribed_constraints_satisfied = true;
        for (const auto& cell : primitive_set.cut_cells) {
            const auto constraint_position =
                wall_constraint_by_parent.find(cell.parent_cell);
            if (constraint_position == wall_constraint_by_parent.end() ||
                constraint_position->second.kind !=
                    LevelSetWallContactConstraintKind::PrescribedAngle) {
                continue;
            }
            const auto& constraint = constraint_position->second;
            const auto frame = prescribedContactFrame(
                constraint, mesh.dimension());
            const auto repaired_fit = fitAffineFieldOnCell(
                cell, dof_points, repaired_coefficients, mesh.dimension());
            const auto input_fit = fitAffineFieldOnCell(
                cell, dof_points, input_coefficients, mesh.dimension());
            if (!repaired_fit.valid) {
                prescribed_constraints_satisfied = false;
                continue;
            }

            Real prescribed_value_scale = Real{1.0};
            Real cell_value_residual = repaired_fit.max_value_residual;
            Real cell_length = Real{0.0};
            for (const GlobalIndex dof : cell.dofs) {
                const auto index = static_cast<std::size_t>(dof);
                cell_value_residual = std::max(
                    cell_value_residual,
                    std::abs(repaired_coefficients[index] -
                             prescribed_contact_target[index]));
                prescribed_value_scale = std::max(
                    {prescribed_value_scale,
                     std::abs(repaired_coefficients[index]),
                     std::abs(prescribed_contact_target[index])});
                for (const GlobalIndex other_dof : cell.dofs) {
                    cell_length = std::max(
                        cell_length,
                        distance(dof_points[index],
                                 dof_points[static_cast<std::size_t>(
                                     other_dof)]));
                }
            }
            result.max_prescribed_contact_value_residual = std::max(
                result.max_prescribed_contact_value_residual,
                cell_value_residual);
            if (!(cell_length > Real{0.0}) ||
                !std::isfinite(cell_length)) {
                prescribed_constraints_satisfied = false;
                continue;
            }
            const Real value_tolerance = std::max(
                Real{64.0} * options.signed_distance_tolerance,
                Real{4096.0} * std::numeric_limits<Real>::epsilon() *
                    prescribed_value_scale);
            const Real gradient_tolerance = std::max(
                Real{4096.0} * std::numeric_limits<Real>::epsilon(),
                value_tolerance / cell_length);
            const Real angle_tolerance = gradient_tolerance;
            const Real contact_tolerance = std::max(
                Real{64.0} * options.signed_distance_tolerance,
                Real{4096.0} * std::numeric_limits<Real>::epsilon() *
                    cell_length);

            const Real gradient_norm = norm(repaired_fit.gradient);
            if (!(gradient_norm > gradient_tolerance) ||
                !std::isfinite(gradient_norm)) {
                prescribed_constraints_satisfied = false;
                continue;
            }
            const auto unit_gradient = scale(
                repaired_fit.gradient, Real{1.0} / gradient_norm);
            const Real actual_angle = std::acos(std::clamp(
                -dot(unit_gradient, frame.wall_normal),
                Real{-1.0},
                Real{1.0}));
            const Real angle_error = std::abs(
                actual_angle - constraint.target_angle_radians);
            result.max_prescribed_contact_angle_error_radians = std::max(
                result.max_prescribed_contact_angle_error_radians,
                angle_error);

            const Real wall_gradient =
                dot(repaired_fit.gradient, frame.wall_normal);
            const auto tangential_gradient = sub(
                repaired_fit.gradient,
                scale(frame.wall_normal, wall_gradient));
            const Real tangential_gradient_norm =
                norm(tangential_gradient);
            const Real contact_value =
                repaired_fit.intercept +
                dot(repaired_fit.gradient,
                    constraint.accepted_contact_point);
            if (!(tangential_gradient_norm > gradient_tolerance) ||
                !std::isfinite(tangential_gradient_norm)) {
                prescribed_constraints_satisfied = false;
                continue;
            }
            const Real contact_displacement =
                std::abs(contact_value) / tangential_gradient_norm;
            result.max_contact_line_displacement = std::max(
                result.max_contact_line_displacement,
                contact_displacement);
            const Real line_alignment = std::abs(
                dot(unit_gradient, frame.line_tangent));

            if (input_fit.valid &&
                norm(input_fit.gradient) > gradient_tolerance) {
                const auto input_normal = scale(
                    input_fit.gradient,
                    Real{1.0} / norm(input_fit.gradient));
                result.max_contact_angle_change_radians = std::max(
                    result.max_contact_angle_change_radians,
                    std::acos(std::clamp(dot(input_normal, unit_gradient),
                                         Real{-1.0},
                                         Real{1.0})));
            }
            prescribed_constraints_satisfied =
                prescribed_constraints_satisfied &&
                cell_value_residual <= value_tolerance &&
                angle_error <= angle_tolerance &&
                contact_displacement <= contact_tolerance &&
                line_alignment <= angle_tolerance;
        }

        result.wall_contact_constraints_satisfied =
            dynamic_constraints_satisfied &&
            prescribed_constraints_satisfied;
        if (!result.wall_contact_constraints_satisfied) {
            result.success = false;
            result.diagnostic =
                "level-set signed-distance repair violated a wall-contact geometry constraint";
            return result;
        }
        // AcceptedDynamicAngle patches remain a positive common scale, so
        // their accepted crossing and unit normal are unchanged.  Prescribed
        // patches instead match the wall-frame target above.
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
        result.max_unconstrained_signed_distance_error <=
            options.signed_distance_tolerance &&
        result.wall_contact_constraints_satisfied;
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
    std::vector<Real>& repaired_coefficients,
    std::span<const LevelSetWallContactConstraint>
        wall_contact_constraints)
{
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

    const auto synchronized = ownerSynchronizedCoefficients(
        level_set_dofs, input_coefficients);
    const auto primitive_set = globalizePrimitiveSet(
        level_set_dofs,
        buildLinearInterfacePrimitives(mesh,
                                       level_set_dofs,
                                       *entity_map,
                                       options.signed_distance_tolerance,
                                       synchronized));
    const auto global_wall_contact_constraints =
        globalizeWallContactConstraints(level_set_dofs,
                                        wall_contact_constraints);
    std::vector<Real> candidate(synchronized.begin(), synchronized.end());
    auto result = repairSignedDistanceCoefficientsFromPrimitives(
        mesh,
        level_set_dofs,
        options,
        synchronized,
        candidate,
        primitive_set,
        global_wall_contact_constraints,
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
    repaired_coefficients = std::move(candidate);
    return result;
}

LevelSetSignedDistanceRepairResult repairLevelSetSignedDistanceByProjection(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetReinitializationOptions& options,
    std::span<const Real> input_solution,
    std::vector<Real>& repaired_solution,
    std::span<const LevelSetWallContactConstraint>
        wall_contact_constraints)
{
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
    field_coefficients = ownerSynchronizedCoefficients(field_dofs,
                                                       field_coefficients);

    std::vector<Real> repaired_field;
    repaired_field.assign(field_coefficients.begin(), field_coefficients.end());
    const auto& mesh_access = system.meshAccess();
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::invalid_argument(
            "level-set signed-distance repair requires a scalar nodal field");
    }
    const auto primitive_set = globalizePrimitiveSet(
        field_dofs,
        buildLinearInterfacePrimitives(mesh_access,
                                       field_dofs,
                                       *entity_map,
                                       options.signed_distance_tolerance,
                                       field_coefficients));
    const auto global_wall_contact_constraints =
        globalizeWallContactConstraints(field_dofs,
                                        wall_contact_constraints);

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        auto result = repairSignedDistanceCoefficientsFromPrimitives(
            mesh_access,
            field_dofs,
            options,
            std::span<const Real>(field_coefficients.data(),
                                  field_coefficients.size()),
            repaired_field,
            primitive_set,
            global_wall_contact_constraints,
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
        options,
        std::span<const Real>(field_coefficients.data(),
                              field_coefficients.size()),
        repaired_field,
        primitive_set,
        global_wall_contact_constraints,
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
        options,
        std::span<const Real>(field_coefficients.data(),
                              field_coefficients.size()),
        repaired_field,
        primitive_set,
        global_wall_contact_constraints,
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
