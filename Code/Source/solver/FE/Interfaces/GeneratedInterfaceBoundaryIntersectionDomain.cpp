/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"

#include "Assembly/Assembler.h"
#include "Basis/NodeOrderingConventions.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace svmp {
namespace FE {
namespace interfaces {
namespace {

constexpr Real kTiny = Real{1.0e-30};

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

[[nodiscard]] std::array<Real, 3> unitOrDefault(
    const std::array<Real, 3>& value,
    const std::array<Real, 3>& fallback) noexcept
{
    const Real n = norm(value);
    if (!(n > kTiny) || !std::isfinite(n)) {
        return fallback;
    }
    return scale(value, Real{1.0} / n);
}

[[nodiscard]] bool samePoint(const std::array<Real, 3>& a,
                             const std::array<Real, 3>& b,
                             Real tolerance) noexcept
{
    return norm(sub(a, b)) <= tolerance;
}

void addUniquePoint(std::vector<std::array<Real, 3>>& points,
                    const std::array<Real, 3>& point,
                    Real tolerance)
{
    const auto duplicate =
        std::find_if(points.begin(), points.end(), [&](const auto& existing) {
            return samePoint(existing, point, tolerance);
        });
    if (duplicate == points.end()) {
        points.push_back(point);
    }
}

[[nodiscard]] std::vector<std::size_t> cornerIndices(ElementType type)
{
    switch (type) {
    case ElementType::Triangle3:
    case ElementType::Triangle6:
        return {0u, 1u, 2u};
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
        return {0u, 1u, 2u, 3u};
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return {0u, 1u, 2u, 3u};
    case ElementType::Hex8:
    case ElementType::Hex20:
    case ElementType::Hex27:
        return {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u};
    case ElementType::Wedge6:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
        return {0u, 1u, 2u, 3u, 4u, 5u};
    case ElementType::Pyramid5:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14:
        return {0u, 1u, 2u, 3u, 4u};
    default:
        return {};
    }
}

[[nodiscard]] std::vector<std::size_t> localFaceCornerIndices(
    ElementType type,
    LocalIndex face)
{
    const auto f = static_cast<int>(face);
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

[[nodiscard]] std::array<Real, 3> centroid(
    const std::vector<std::array<Real, 3>>& points,
    const std::vector<std::size_t>& indices)
{
    std::array<Real, 3> c{{0.0, 0.0, 0.0}};
    if (indices.empty()) {
        return c;
    }
    for (const auto i : indices) {
        if (i < points.size()) {
            c = add(c, points[i]);
        }
    }
    return scale(c, Real{1.0} / static_cast<Real>(indices.size()));
}

[[nodiscard]] std::array<Real, 3> outwardNormal(
    int dimension,
    const std::vector<std::array<Real, 3>>& cell_points,
    const std::vector<std::size_t>& cell_corners,
    const std::vector<std::size_t>& face_corners)
{
    if (face_corners.size() < 2u) {
        return {{0.0, 1.0, 0.0}};
    }
    const auto cell_center = centroid(cell_points, cell_corners);
    const auto face_center = centroid(cell_points, face_corners);
    std::array<Real, 3> n{{0.0, 1.0, 0.0}};
    if (dimension == 2) {
        const auto edge = sub(cell_points[face_corners[1]], cell_points[face_corners[0]]);
        n = unitOrDefault({{edge[1], -edge[0], 0.0}}, {{0.0, 1.0, 0.0}});
    } else if (face_corners.size() >= 3u) {
        const auto e0 = sub(cell_points[face_corners[1]], cell_points[face_corners[0]]);
        const auto e1 = sub(cell_points[face_corners[2]], cell_points[face_corners[0]]);
        n = unitOrDefault(cross(e0, e1), {{0.0, 0.0, 1.0}});
    }
    if (dot(n, sub(face_center, cell_center)) < Real{0.0}) {
        n = scale(n, Real{-1.0});
    }
    return n;
}

[[nodiscard]] bool pointOnSegment(const std::array<Real, 3>& point,
                                  const std::array<Real, 3>& a,
                                  const std::array<Real, 3>& b,
                                  Real tolerance) noexcept
{
    const auto ab = sub(b, a);
    const auto ap = sub(point, a);
    const Real ab2 = dot(ab, ab);
    if (!(ab2 > kTiny)) {
        return samePoint(point, a, tolerance);
    }
    const Real t = dot(ap, ab) / ab2;
    if (t < -tolerance || t > Real{1.0} + tolerance) {
        return false;
    }
    const auto closest = add(a, scale(ab, std::clamp(t, Real{0.0}, Real{1.0})));
    return samePoint(point, closest, tolerance);
}

[[nodiscard]] bool pointOnFace(const std::array<Real, 3>& point,
                               const std::vector<std::array<Real, 3>>& cell_points,
                               const std::vector<std::size_t>& face_corners,
                               const std::array<Real, 3>& normal,
                               Real tolerance) noexcept
{
    if (face_corners.size() < 3u) {
        return false;
    }
    const auto& origin = cell_points[face_corners[0]];
    if (std::abs(dot(sub(point, origin), normal)) > tolerance) {
        return false;
    }
    const auto center = centroid(cell_points, face_corners);
    for (std::size_t i = 0; i < face_corners.size(); ++i) {
        const auto& a = cell_points[face_corners[i]];
        const auto& b = cell_points[face_corners[(i + 1u) % face_corners.size()]];
        const auto edge = sub(b, a);
        const auto inward = cross(normal, edge);
        const Real ref = dot(inward, sub(center, a));
        const Real value = dot(inward, sub(point, a));
        if (ref >= Real{0.0}) {
            if (value < -tolerance) {
                return false;
            }
        } else if (value > tolerance) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::uint64_t stableFragmentId(
    const GeneratedInterfaceBoundaryIntersectionFragment& fragment,
    std::uint64_t source_revision) noexcept
{
    std::uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](std::uint64_t value) noexcept {
        h ^= value;
        h *= 1099511628211ull;
    };
    mix(static_cast<std::uint64_t>(fragment.intersection_marker));
    mix(static_cast<std::uint64_t>(fragment.interface_marker));
    mix(static_cast<std::uint64_t>(fragment.boundary_marker));
    mix(static_cast<std::uint64_t>(
        fragment.parent_cell_global_id != INVALID_GLOBAL_INDEX
            ? fragment.parent_cell_global_id
            : static_cast<GlobalIndex>(fragment.parent_cell)));
    mix(static_cast<std::uint64_t>(
        fragment.parent_face_global_id != INVALID_GLOBAL_INDEX
            ? fragment.parent_face_global_id
            : static_cast<GlobalIndex>(fragment.parent_face)));
    mix(static_cast<std::uint64_t>(fragment.local_fragment_index));
    mix(fragment.source_interface_stable_id);
    mix(source_revision);
    return h;
}

[[nodiscard]] geometry::CutQuadratureRule toRule(
    const GeneratedInterfaceBoundaryIntersectionRequest& request,
    const GeneratedInterfaceBoundaryIntersectionFragment& fragment)
{
    geometry::CutQuadratureRule rule;
    rule.kind = geometry::CutQuadratureKind::Interface;
    rule.side = geometry::CutIntegrationSide::Interface;
    rule.geometric_dimension =
        fragment.kind == GeneratedInterfaceBoundaryIntersectionKind::Point ? 0 : 1;
    rule.measure = fragment.measure;
    rule.exact_for_constants = true;
    const int supported_rule_order =
        fragment.quadrature_points.size() <= 1u
            ? 1
            : (fragment.quadrature_points.size() == 2u ? 3 : 5);
    const int achieved_order =
        fragment.kind == GeneratedInterfaceBoundaryIntersectionKind::Point
            ? 0
            : std::min(request.quadrature_order, supported_rule_order);
    rule.exact_polynomial_order = achieved_order;
    rule.policy.kind = geometry::CutQuadratureConstructionKind::TopologySubdivision;
    rule.policy.polynomial_order = achieved_order;
    rule.policy.name = fragment.kind == GeneratedInterfaceBoundaryIntersectionKind::Point
                           ? "generated-interface-boundary-point-trace"
                           : "generated-interface-boundary-segment-trace";
    rule.provenance.embedded_geometry_id = request.source.identifier();
    rule.provenance.cut_topology_id = fragment.topology_id;
    rule.provenance.parent_entity = fragment.parent_cell;
    rule.provenance.parent_boundary_entity = fragment.parent_face;
    rule.provenance.parent_entity_global_id =
        fragment.parent_cell_global_id;
    rule.provenance.parent_boundary_entity_global_id =
        fragment.parent_face_global_id;
    rule.provenance.owner_rank = fragment.owner_rank;
    rule.provenance.marker = request.resolvedIntersectionMarker();
    rule.provenance.cut_topology_revision = fragment.stable_id;
    rule.provenance.predicate_policy_key = request.quadrature_policy_key;
    rule.provenance.source_value_revision =
        request.source_value_revision != 0u ? request.source_value_revision
                                            : request.source.value_revision;
    rule.provenance.source_stable_id =
        fragment.source_interface_stable_id;
    rule.provenance.construction = rule.policy.kind;
    rule.provenance.frame = geometry::CutGeometryFrame::Reference;
    rule.provenance.implicit_geometry_mode =
        fragment.represented_implicit_geometry_mode;
    rule.provenance.implicit_quadrature_backend =
        fragment.represented_implicit_quadrature_backend;
    rule.provenance.selected_implicit_quadrature_backend =
        fragment.represented_implicit_quadrature_backend;
    rule.provenance.implicit_fallback_status =
        fragment.represented_implicit_fallback_status;
    rule.provenance.requested_quadrature_order = request.quadrature_order;
    rule.provenance.achieved_quadrature_order = achieved_order;
    rule.provenance_id = request.source.identifier();
    rule.frame = geometry::CutGeometryFrame::Reference;
    rule.points.reserve(fragment.quadrature_points.size());
    for (const auto& point : fragment.quadrature_points) {
        geometry::CutQuadraturePoint qp;
        qp.point = point.point;
        qp.parent_coordinate = point.parent_coordinate;
        qp.normal = point.interface_normal;
        qp.boundary_normal = point.boundary_normal;
        qp.tangent = point.tangent;
        qp.weight = point.weight;
        qp.reference_measure_factor =
            point.reference_measure_factor > Real{0.0}
                ? point.reference_measure_factor
                : point.weight;
        qp.level_set_residual = point.level_set_residual;
        qp.gradient_norm = point.gradient_norm;
        rule.points.push_back(qp);
    }
    return rule;
}

struct BoundaryFaceInfo {
    MeshIndex face_id{static_cast<MeshIndex>(-1)};
    LocalIndex local_face{INVALID_LOCAL_INDEX};
    int marker{-1};
    std::vector<GlobalIndex> cell_nodes{};
    std::vector<std::array<Real, 3>> cell_points{};
    std::vector<std::array<Real, 3>> cell_reference_points{};
    std::vector<std::size_t> cell_corners{};
    std::vector<std::size_t> face_corners{};
    std::array<Real, 3> normal{{0.0, 1.0, 0.0}};
    std::array<Real, 3> tangent{{1.0, 0.0, 0.0}};
};

[[nodiscard]] std::unordered_map<MeshIndex, std::vector<BoundaryFaceInfo>>
collectBoundaryFaces(const assembly::IMeshAccess& mesh, int boundary_marker)
{
    std::unordered_map<MeshIndex, std::vector<BoundaryFaceInfo>> faces_by_cell;
    std::unordered_set<GlobalIndex> seen_owned_faces;
    mesh.forEachBoundaryFace(
        boundary_marker,
        [&](GlobalIndex face_id, GlobalIndex cell_id) {
            if (!mesh.isOwnedCell(cell_id)) {
                return;
            }
            if (!seen_owned_faces.insert(face_id).second) {
                return;
            }
            const auto type = mesh.getCellType(cell_id);
            BoundaryFaceInfo info;
            info.face_id = static_cast<MeshIndex>(face_id);
            info.local_face = mesh.getLocalFaceIndex(face_id, cell_id);
            info.marker = mesh.getBoundaryFaceMarker(face_id);
            info.cell_corners = cornerIndices(type);
            info.face_corners = localFaceCornerIndices(type, info.local_face);
            if (info.face_corners.size() < 2u || info.cell_corners.empty()) {
                return;
            }
            mesh.getCellCoordinates(cell_id, info.cell_points);
            if (info.cell_points.empty()) {
                return;
            }
            mesh.getCellNodes(cell_id, info.cell_nodes);
            info.cell_reference_points.reserve(info.cell_points.size());
            for (std::size_t node = 0; node < info.cell_points.size(); ++node) {
                const auto xi =
                    basis::ReferenceNodeLayout::get_node_coords(type, node);
                info.cell_reference_points.push_back(
                    {{xi[0], xi[1], xi[2]}});
            }
            info.normal =
                outwardNormal(mesh.dimension(), info.cell_reference_points,
                              info.cell_corners, info.face_corners);
            const auto edge =
                sub(info.cell_reference_points[info.face_corners[1]],
                    info.cell_reference_points[info.face_corners[0]]);
            info.tangent = unitOrDefault(edge, {{1.0, 0.0, 0.0}});
            faces_by_cell[static_cast<MeshIndex>(cell_id)].push_back(std::move(info));
        });
    return faces_by_cell;
}

[[nodiscard]] GeneratedInterfaceBoundaryIntersectionFragment skippedFragment(
    const GeneratedInterfaceBoundaryIntersectionRequest& request,
    MeshIndex parent_cell,
    MeshIndex parent_face,
    GeneratedInterfaceBoundaryIntersectionDegeneracy degeneracy,
    std::string diagnostic)
{
    GeneratedInterfaceBoundaryIntersectionFragment fragment;
    fragment.interface_marker = request.interface_marker;
    fragment.boundary_marker = request.boundary_marker;
    fragment.intersection_marker = request.resolvedIntersectionMarker();
    fragment.parent_cell = parent_cell;
    fragment.parent_face = parent_face;
    fragment.degeneracy = degeneracy;
    fragment.diagnostic = std::move(diagnostic);
    return fragment;
}

} // namespace

std::string GeneratedInterfaceBoundaryIntersectionMarkerKey::stableKey() const
{
    const Real canonical_isovalue =
        isovalue == Real{0.0}
            ? Real{0.0}
            : isovalue;
    std::ostringstream encoded_isovalue;
    encoded_isovalue.imbue(
        std::locale::classic());
    encoded_isovalue
        << std::scientific
        << std::setprecision(
               std::numeric_limits<
                   Real>::max_digits10)
        << canonical_isovalue;

    std::string key;
    const auto append_component =
        [&](std::string_view value) {
            key += std::to_string(
                value.size());
            key.push_back(':');
            key.append(value);
        };
    append_component(
        std::to_string(
            static_cast<int>(
                source.kind)));
    if (source.kind ==
        CutInterfaceSourceKind::Field) {
        append_component(
            std::to_string(
                source.field_id));
    } else {
        append_component(
            source.evaluator_id);
    }
    append_component(domain_id);
    append_component(
        encoded_isovalue.str());
    append_component(
        std::to_string(interface_marker));
    append_component(
        std::to_string(boundary_marker));
    return key;
}

std::uint64_t stableGeneratedInterfaceBoundaryIntersectionMarkerHash(
    const GeneratedInterfaceBoundaryIntersectionMarkerKey& key)
{
    std::uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](std::uint64_t value) noexcept {
        h ^= value;
        h *= 1099511628211ull;
    };
    const auto stable_key = key.stableKey();
    for (const char c : stable_key) {
        mix(static_cast<unsigned char>(c));
    }
    return h;
}

int stableGeneratedInterfaceBoundaryIntersectionMarker(
    const GeneratedInterfaceBoundaryIntersectionMarkerKey& key,
    int marker_base,
    int marker_range)
{
    if (key.requested_marker >= 0) {
        return key.requested_marker;
    }
    if (marker_base < 0 || marker_range <= 0) {
        throw std::invalid_argument(
            "generated interface-boundary marker range must be positive");
    }
    const auto offset =
        static_cast<int>(
            stableGeneratedInterfaceBoundaryIntersectionMarkerHash(key) %
            static_cast<std::uint64_t>(marker_range));
    return marker_base + offset;
}

bool GeneratedInterfaceBoundaryIntersectionRequest::valid() const noexcept
{
    return source.valid() && interface_marker >= 0 && boundary_marker >= 0 &&
           std::isfinite(isovalue) && std::isfinite(tolerance) &&
           tolerance > Real{0.0} && quadrature_order >= 0 &&
           frame == geometry::CutGeometryFrame::Reference;
}

int GeneratedInterfaceBoundaryIntersectionRequest::resolvedIntersectionMarker()
    const
{
    GeneratedInterfaceBoundaryIntersectionMarkerKey key;
    key.source = source;
    key.domain_id = generated_domain_id;
    key.isovalue = isovalue;
    key.interface_marker = interface_marker;
    key.boundary_marker = boundary_marker;
    key.requested_marker = intersection_marker;
    return stableGeneratedInterfaceBoundaryIntersectionMarker(key);
}

bool GeneratedInterfaceBoundaryIntersectionFragment::active() const noexcept
{
    return intersection_marker >= 0 && interface_marker >= 0 &&
           boundary_marker >= 0 && parent_cell >= static_cast<MeshIndex>(0) &&
           parent_face >= static_cast<MeshIndex>(0) && measure > Real{0.0} &&
           !quadrature_points.empty() &&
           (degeneracy == GeneratedInterfaceBoundaryIntersectionDegeneracy::None ||
            degeneracy ==
                GeneratedInterfaceBoundaryIntersectionDegeneracy::FallbackRule);
}

GeneratedInterfaceBoundaryIntersectionDomain::
    GeneratedInterfaceBoundaryIntersectionDomain(
        GeneratedInterfaceBoundaryIntersectionRequest request)
    : request_(std::move(request))
{
}

const GeneratedInterfaceBoundaryIntersectionRequest&
GeneratedInterfaceBoundaryIntersectionDomain::request() const noexcept
{
    return request_;
}

int GeneratedInterfaceBoundaryIntersectionDomain::marker() const noexcept
{
    return request_.resolvedIntersectionMarker();
}

int GeneratedInterfaceBoundaryIntersectionDomain::boundaryMarker() const noexcept
{
    return request_.boundary_marker;
}

bool GeneratedInterfaceBoundaryIntersectionDomain::empty() const noexcept
{
    return fragments_.empty();
}

const std::vector<GeneratedInterfaceBoundaryIntersectionFragment>&
GeneratedInterfaceBoundaryIntersectionDomain::fragments() const noexcept
{
    return fragments_;
}

void GeneratedInterfaceBoundaryIntersectionDomain::addFragment(
    GeneratedInterfaceBoundaryIntersectionFragment fragment)
{
    if (fragment.interface_marker < 0) {
        fragment.interface_marker = request_.interface_marker;
    }
    if (fragment.boundary_marker < 0) {
        fragment.boundary_marker = request_.boundary_marker;
    }
    if (fragment.intersection_marker < 0) {
        fragment.intersection_marker = request_.resolvedIntersectionMarker();
    }
    if (fragment.local_fragment_index == INVALID_LOCAL_INDEX) {
        fragment.local_fragment_index =
            static_cast<LocalIndex>(fragments_.size());
    }
    if (fragment.stable_id == 0u) {
        fragment.stable_id =
            stableFragmentId(fragment,
                             request_.source_value_revision != 0u
                                 ? request_.source_value_revision
                                 : request_.source.value_revision);
    }
    if (fragment.topology_id.empty()) {
        const auto cell_identity =
            fragment.parent_cell_global_id != INVALID_GLOBAL_INDEX
                ? fragment.parent_cell_global_id
                : static_cast<GlobalIndex>(fragment.parent_cell);
        const auto face_identity =
            fragment.parent_face_global_id != INVALID_GLOBAL_INDEX
                ? fragment.parent_face_global_id
                : static_cast<GlobalIndex>(fragment.parent_face);
        fragment.topology_id =
            "interface_boundary:" + std::to_string(fragment.interface_marker) +
            ":" + std::to_string(fragment.boundary_marker) + ":" +
            std::to_string(cell_identity) + ":" +
            std::to_string(face_identity) + ":" +
            std::to_string(fragment.local_fragment_index);
    }
    fragments_.push_back(std::move(fragment));
}

void GeneratedInterfaceBoundaryIntersectionDomain::addSkippedFragment(
    GeneratedInterfaceBoundaryIntersectionFragment fragment)
{
    addFragment(std::move(fragment));
}

GeneratedInterfaceBoundaryIntersectionSummary
GeneratedInterfaceBoundaryIntersectionDomain::summary() const noexcept
{
    GeneratedInterfaceBoundaryIntersectionSummary s;
    s.interface_marker = request_.interface_marker;
    s.boundary_marker = request_.boundary_marker;
    s.intersection_marker = request_.resolvedIntersectionMarker();
    s.fragment_count = fragments_.size();
    bool saw_weight = false;
    s.min_weight = std::numeric_limits<Real>::infinity();
    s.max_weight = -std::numeric_limits<Real>::infinity();
    for (const auto& fragment : fragments_) {
        if (!fragment.active()) {
            ++s.skipped_fragment_count;
        }
        if (fragment.degeneracy ==
            GeneratedInterfaceBoundaryIntersectionDegeneracy::FallbackRule) {
            ++s.fallback_fragment_count;
        } else if (fragment.degeneracy ==
                   GeneratedInterfaceBoundaryIntersectionDegeneracy::
                       TangentIntersection) {
            ++s.tangent_intersection_count;
        } else if (fragment.degeneracy ==
                   GeneratedInterfaceBoundaryIntersectionDegeneracy::
                       AlignedTopology) {
            ++s.aligned_topology_count;
        } else if (fragment.degeneracy ==
                   GeneratedInterfaceBoundaryIntersectionDegeneracy::
                       AmbiguousTopology) {
            ++s.ambiguous_topology_count;
        } else if (fragment.degeneracy ==
                   GeneratedInterfaceBoundaryIntersectionDegeneracy::
                       NearZeroInterfaceGradient) {
            ++s.near_zero_gradient_count;
        } else if (fragment.degeneracy ==
                   GeneratedInterfaceBoundaryIntersectionDegeneracy::
                       VanishingMeasure) {
            ++s.vanishing_measure_count;
        }
        if (!fragment.active()) {
            continue;
        }
        ++s.active_fragment_count;
        s.measure += fragment.measure;
        s.measure_by_boundary_marker[fragment.boundary_marker] += fragment.measure;
        s.quadrature_point_count += fragment.quadrature_points.size();
        for (const auto& point : fragment.quadrature_points) {
            saw_weight = true;
            s.min_weight = std::min(s.min_weight, point.weight);
            s.max_weight = std::max(s.max_weight, point.weight);
        }
    }
    if (!saw_weight) {
        s.min_weight = Real{0.0};
        s.max_weight = Real{0.0};
    }
    return s;
}

std::vector<geometry::CutQuadratureRule>
GeneratedInterfaceBoundaryIntersectionDomain::intersectionQuadratureRules()
    const
{
    std::vector<geometry::CutQuadratureRule> rules;
    rules.reserve(fragments_.size());
    for (const auto& fragment : fragments_) {
        if (fragment.active()) {
            rules.push_back(toRule(request_, fragment));
        }
    }
    std::sort(rules.begin(), rules.end(), cutQuadratureRuleDeterministicLess);
    return rules;
}

GeneratedInterfaceBoundaryIntersectionDomain
buildGeneratedInterfaceBoundaryIntersectionDomain(
    GeneratedInterfaceBoundaryIntersectionRequest request,
    const LevelSetInterfaceDomain& interface_domain,
    const assembly::IMeshAccess& mesh)
{
    if (!request.valid()) {
        throw std::invalid_argument(
            "generated interface-boundary intersection request is invalid");
    }
    GeneratedInterfaceBoundaryIntersectionDomain domain(std::move(request));
    const auto& req = domain.request();
    if (mesh.parallelSize() > 1 && !mesh.globalEntityIdsAvailable()) {
        throw std::invalid_argument(
            "distributed contact geometry requires globally unique cell and face ids");
    }
    const auto faces_by_cell = collectBoundaryFaces(mesh, req.boundary_marker);
    if (faces_by_cell.empty()) {
        return domain;
    }

    for (const auto& fragment : interface_domain.fragments()) {
        if (!fragment.active()) {
            continue;
        }
        const auto faces_it = faces_by_cell.find(fragment.parent_cell);
        if (faces_it == faces_by_cell.end()) {
            continue;
        }
        for (const auto& face : faces_it->second) {
            if (face.face_corners.size() < 2u) {
                auto skipped = skippedFragment(
                    req,
                    fragment.parent_cell,
                    face.face_id,
                    GeneratedInterfaceBoundaryIntersectionDegeneracy::
                        UnsupportedHighOrder,
                    "the authoritative interface fragment cannot be restricted to this boundary topology");
                skipped.parent_cell_global_id =
                    mesh.globalEntityIdsAvailable()
                        ? mesh.getCellGlobalId(fragment.parent_cell)
                        : static_cast<GlobalIndex>(fragment.parent_cell);
                skipped.parent_face_global_id =
                    mesh.globalEntityIdsAvailable()
                        ? mesh.getBoundaryFaceGlobalId(face.face_id)
                        : static_cast<GlobalIndex>(face.face_id);
                skipped.owner_rank = mesh.getBoundaryFaceOwnerRank(
                    face.face_id, fragment.parent_cell);
                domain.addSkippedFragment(std::move(skipped));
                continue;
            }

            std::vector<std::array<Real, 3>> points_on_face;
            if (mesh.dimension() == 2) {
                const auto& a =
                    face.cell_reference_points[face.face_corners[0]];
                const auto& b =
                    face.cell_reference_points[face.face_corners[1]];
                for (const auto& vertex : fragment.vertices) {
                    if (pointOnSegment(vertex.parent_coordinate,
                                       a,
                                       b,
                                       req.tolerance)) {
                        addUniquePoint(points_on_face,
                                       vertex.parent_coordinate,
                                       req.tolerance);
                    }
                }
            } else {
                for (const auto& vertex : fragment.vertices) {
                    if (pointOnFace(vertex.parent_coordinate,
                                    face.cell_reference_points,
                                    face.face_corners,
                                    face.normal,
                                    req.tolerance)) {
                        addUniquePoint(points_on_face,
                                       vertex.parent_coordinate,
                                       req.tolerance);
                    }
                }
            }

            if (points_on_face.empty()) {
                continue;
            }

            GeneratedInterfaceBoundaryIntersectionFragment out;
            out.interface_marker = req.interface_marker;
            out.boundary_marker = req.boundary_marker;
            out.intersection_marker = req.resolvedIntersectionMarker();
            out.parent_cell = fragment.parent_cell;
            out.parent_face = face.face_id;
            out.parent_cell_global_id =
                mesh.globalEntityIdsAvailable()
                    ? mesh.getCellGlobalId(fragment.parent_cell)
                    : static_cast<GlobalIndex>(fragment.parent_cell);
            out.parent_face_global_id =
                mesh.globalEntityIdsAvailable()
                    ? mesh.getBoundaryFaceGlobalId(face.face_id)
                    : static_cast<GlobalIndex>(face.face_id);
            out.owner_rank = mesh.getBoundaryFaceOwnerRank(
                face.face_id, fragment.parent_cell);
            if (fragment.parent_cell_global_id != INVALID_GLOBAL_INDEX &&
                fragment.parent_cell_global_id != out.parent_cell_global_id) {
                throw std::invalid_argument(
                    "generated contact geometry source surface has a stale global parent-cell id");
            }
            if (fragment.owner_rank >= 0 &&
                fragment.owner_rank != mesh.getCellOwnerRank(fragment.parent_cell)) {
                throw std::invalid_argument(
                    "generated contact geometry source surface has stale owner metadata");
            }
            out.source_interface_stable_id = fragment.stable_id;
            out.represented_implicit_geometry_mode =
                interface_domain.request().implicit_geometry_mode;
            out.represented_implicit_quadrature_backend =
                fragment.implicit_quadrature_backend.empty()
                    ? interface_domain.request().implicit_quadrature_backend
                    : fragment.implicit_quadrature_backend;
            out.represented_implicit_fallback_status =
                fragment.implicit_fallback_status.empty()
                    ? interface_domain.request().implicit_fallback_status
                    : fragment.implicit_fallback_status;
            if (out.represented_implicit_geometry_mode.empty() ||
                out.represented_implicit_quadrature_backend.empty() ||
                out.represented_implicit_quadrature_backend == "Auto" ||
                out.represented_implicit_quadrature_backend == "Unknown") {
                throw std::invalid_argument(
                    "generated contact geometry requires an explicit represented implicit backend");
            }
            out.interface_normal = fragment.normal;
            out.boundary_normal = face.normal;
            out.tangent = face.tangent;
            // This is the exact boundary trace of the already-selected
            // generated surface fragment, not an independently reconstructed
            // fallback.  Retaining the fragment normal here makes dI and its
            // d-2 boundary use one discrete geometry and surface energy.
            out.degeneracy =
                GeneratedInterfaceBoundaryIntersectionDegeneracy::None;

            if (mesh.dimension() == 2) {
                if (points_on_face.size() > 1u) {
                    out.degeneracy =
                        GeneratedInterfaceBoundaryIntersectionDegeneracy::
                            TangentIntersection;
                    out.diagnostic =
                        "generated interface overlaps the marked boundary edge";
                    if (!req.keep_degenerate_fragments) {
                        out.measure = Real{0.0};
                        domain.addSkippedFragment(std::move(out));
                        continue;
                    }
                }
                const auto point = points_on_face.front();
                out.kind = GeneratedInterfaceBoundaryIntersectionKind::Point;
                out.measure = Real{1.0};
                out.tangent = face.tangent;
                out.vertices.push_back(point);
                out.quadrature_points.push_back(
                    GeneratedInterfaceBoundaryIntersectionQuadraturePoint{
                        .point = point,
                        .parent_coordinate = point,
                        .interface_normal = fragment.normal,
                        .boundary_normal = face.normal,
                        .tangent = out.tangent,
                        .weight = out.measure,
                        .reference_measure_factor = out.measure,
                        .gradient_norm = fragment.min_gradient_norm});
                domain.addFragment(std::move(out));
                continue;
            }

            if (points_on_face.size() < 2u) {
                out.degeneracy =
                    GeneratedInterfaceBoundaryIntersectionDegeneracy::
                        VanishingMeasure;
                out.diagnostic =
                    "generated interface touches the marked boundary face at fewer than two distinct points";
                domain.addSkippedFragment(std::move(out));
                continue;
            }

            std::pair<std::size_t, std::size_t> pair{0u, 1u};
            Real max_length = Real{-1.0};
            for (std::size_t i = 0; i < points_on_face.size(); ++i) {
                for (std::size_t j = i + 1u; j < points_on_face.size(); ++j) {
                    const Real length = norm(sub(points_on_face[j], points_on_face[i]));
                    if (length > max_length) {
                        max_length = length;
                        pair = {i, j};
                    }
                }
            }
            if (!(max_length > req.tolerance)) {
                out.degeneracy =
                    GeneratedInterfaceBoundaryIntersectionDegeneracy::
                        VanishingMeasure;
                out.diagnostic =
                    "generated interface-boundary segment has vanishing measure";
                domain.addSkippedFragment(std::move(out));
                continue;
            }
            const auto& p0 = points_on_face[pair.first];
            const auto& p1 = points_on_face[pair.second];
            out.kind = GeneratedInterfaceBoundaryIntersectionKind::Segment;
            out.measure = max_length;
            out.tangent = unitOrDefault(sub(p1, p0), face.tangent);
            out.vertices = {p0, p1};
            const auto add_segment_point = [&](Real t, Real weight_fraction) {
                const auto point = add(scale(p0, Real{1.0} - t), scale(p1, t));
                out.quadrature_points.push_back(
                    GeneratedInterfaceBoundaryIntersectionQuadraturePoint{
                        .point = point,
                        .parent_coordinate = point,
                        .interface_normal = fragment.normal,
                        .boundary_normal = face.normal,
                        .tangent = out.tangent,
                        .weight = out.measure * weight_fraction,
                        .reference_measure_factor = out.measure,
                        .gradient_norm = fragment.min_gradient_norm});
            };
            if (req.quadrature_order <= 1) {
                add_segment_point(Real{0.5}, Real{1.0});
            } else if (req.quadrature_order <= 3) {
                constexpr Real offset = Real{0.28867513459481288225};
                add_segment_point(Real{0.5} - offset, Real{0.5});
                add_segment_point(Real{0.5} + offset, Real{0.5});
            } else {
                constexpr Real offset = Real{0.38729833462074168852};
                add_segment_point(Real{0.5} - offset, Real{5.0} / Real{18.0});
                add_segment_point(Real{0.5}, Real{4.0} / Real{9.0});
                add_segment_point(Real{0.5} + offset, Real{5.0} / Real{18.0});
            }
            domain.addFragment(std::move(out));
        }
    }

    return domain;
}

GeneratedInterfaceBoundaryIntersectionDomain
buildGeneratedInterfaceBoundaryIntersectionDomain(
    GeneratedInterfaceBoundaryIntersectionRequest request,
    const LevelSetInterfaceDomain& interface_domain,
    const assembly::IMeshAccess& mesh,
    const GeneratedInterfaceBoundaryIntersectionScalarField& scalar_field)
{
    // The caller synchronizes this scalar field for diagnostics, but contact
    // geometry has one owner: the retained interface fragments. Rebuilding
    // roots independently from nodal values can create contact points with no
    // source surface fragment, so both overloads intentionally share the
    // authoritative fragment-trace construction.
    (void)scalar_field;
    return buildGeneratedInterfaceBoundaryIntersectionDomain(
        std::move(request), interface_domain, mesh);
}

GeneratedInterfaceBoundaryProvenanceSummary
validateGeneratedInterfaceBoundaryProvenance(
    const GeneratedInterfaceBoundaryIntersectionDomain& contact_domain,
    const LevelSetInterfaceDomain& interface_domain)
{
    GeneratedInterfaceBoundaryProvenanceSummary summary;
    const auto& contact_request = contact_domain.request();
    const auto& interface_request = interface_domain.request();
    if (contact_request.interface_marker != interface_domain.marker() ||
        contact_request.source.identifier() !=
            interface_request.source.identifier() ||
        contact_request.source.layout_revision !=
            interface_request.source.layout_revision ||
        contact_request.mesh_geometry_revision !=
            interface_request.mesh_geometry_revision ||
        contact_request.mesh_topology_revision !=
            interface_request.mesh_topology_revision ||
        contact_request.ownership_revision !=
            interface_request.ownership_revision ||
        contact_request.quadrature_policy_key !=
            interface_request.quadrature_policy_key ||
        contact_request.source_value_revision !=
            interface_request.source.value_revision) {
        ++summary.stale_revision_count;
        throw std::invalid_argument(
            "generated contact geometry and source interface do not share one revision key");
    }

    std::unordered_map<std::uint64_t, const CutInterfaceFragment*>
        source_by_id;
    for (const auto& fragment : interface_domain.fragments()) {
        if (!fragment.active()) {
            continue;
        }
        ++summary.source_surface_fragment_count;
        if (fragment.stable_id == 0u ||
            !source_by_id.emplace(fragment.stable_id, &fragment).second) {
            ++summary.duplicate_source_surface_id_count;
        }
    }
    if (summary.duplicate_source_surface_id_count != 0u) {
        throw std::invalid_argument(
            "generated interface contains duplicate or zero stable fragment identifiers");
    }

    std::unordered_set<std::uint64_t> referenced_source_ids;
    const auto finite_point = [](const std::array<Real, 3>& point) noexcept {
        return std::isfinite(point[0]) && std::isfinite(point[1]) &&
               std::isfinite(point[2]);
    };
    const auto unit_vector = [](const std::array<Real, 3>& vector) noexcept {
        const Real length2 = vector[0] * vector[0] +
                             vector[1] * vector[1] +
                             vector[2] * vector[2];
        return std::isfinite(length2) &&
               std::abs(length2 - Real{1.0}) <= Real{1.0e-8};
    };
    for (const auto& contact : contact_domain.fragments()) {
        if (!contact.active()) {
            continue;
        }
        ++summary.active_contact_fragment_count;
        const auto source = source_by_id.find(
            contact.source_interface_stable_id);
        if (source == source_by_id.end() ||
            source->second->parent_cell != contact.parent_cell ||
            (source->second->parent_cell_global_id != INVALID_GLOBAL_INDEX &&
             contact.parent_cell_global_id !=
                 source->second->parent_cell_global_id) ||
            (source->second->owner_rank >= 0 &&
             contact.owner_rank != source->second->owner_rank) ||
            source->second->interface_marker != contact.interface_marker) {
            ++summary.orphan_contact_fragment_count;
            continue;
        }
        const std::string source_backend =
            source->second->implicit_quadrature_backend.empty()
                ? interface_request.implicit_quadrature_backend
                : source->second->implicit_quadrature_backend;
        const std::string source_fallback =
            source->second->implicit_fallback_status.empty()
                ? interface_request.implicit_fallback_status
                : source->second->implicit_fallback_status;
        if (contact.represented_implicit_geometry_mode !=
                interface_request.implicit_geometry_mode ||
            contact.represented_implicit_quadrature_backend !=
                source_backend ||
            contact.represented_implicit_fallback_status !=
                source_fallback) {
            throw std::invalid_argument(
                "generated contact geometry does not preserve its source surface representation provenance");
        }
        referenced_source_ids.insert(contact.source_interface_stable_id);
        if (!unit_vector(contact.interface_normal) ||
            !unit_vector(contact.boundary_normal) ||
            !unit_vector(contact.tangent)) {
            throw std::invalid_argument(
                "generated contact geometry contains a non-unit direction");
        }
        const std::size_t expected_vertices =
            contact.kind ==
                    GeneratedInterfaceBoundaryIntersectionKind::Point
                ? 1u
                : 2u;
        if (contact.vertices.size() != expected_vertices ||
            !std::all_of(contact.vertices.begin(),
                         contact.vertices.end(),
                         finite_point)) {
            throw std::invalid_argument(
                "generated contact geometry contains an invalid authoritative trace vertex set");
        }
        for (const auto& point : contact.quadrature_points) {
            if (!finite_point(point.parent_coordinate) ||
                !finite_point(point.point) || !std::isfinite(point.weight) ||
                !(point.weight > Real{0.0}) ||
                !std::isfinite(point.level_set_residual)) {
                throw std::invalid_argument(
                    "generated contact geometry contains an invalid quadrature point");
            }
            summary.max_level_set_residual = std::max(
                summary.max_level_set_residual,
                std::abs(point.level_set_residual));
        }
    }
    summary.referenced_source_surface_fragment_count =
        referenced_source_ids.size();
    if (summary.orphan_contact_fragment_count != 0u) {
        throw std::invalid_argument(
            "generated contact geometry contains an orphan source-surface reference");
    }
    const Real residual_tolerance =
        std::max(contact_request.tolerance, Real{1.0e-14});
    if (summary.max_level_set_residual > residual_tolerance) {
        throw std::invalid_argument(
            "generated contact geometry exceeds its level-set root tolerance");
    }
    return summary;
}

std::vector<int> boundaryMarkers(const assembly::IMeshAccess& mesh)
{
    std::vector<int> markers;
    mesh.forEachBoundaryFace(-1, [&](GlobalIndex face_id, GlobalIndex /*cell_id*/) {
        const int marker = mesh.getBoundaryFaceMarker(face_id);
        if (marker >= 0 &&
            std::find(markers.begin(), markers.end(), marker) == markers.end()) {
            markers.push_back(marker);
        }
    });
    std::sort(markers.begin(), markers.end());
    return markers;
}

} // namespace interfaces
} // namespace FE
} // namespace svmp
