#include "Interfaces/GeneratedActiveBoundaryDomain.h"

#include "Basis/NodeOrderingConventions.h"
#include "Interfaces/detail/ProducerArithmeticAssessment.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace svmp::FE::interfaces {
namespace {

using Point = std::array<Real, 3>;

// Parent-local identities never escape this face construction.
struct PointOrigin {
    enum class Kind { Unknown, Corner, EdgeRoot };
    Kind kind{Kind::Unknown};
    std::size_t first{0u}, second{0u};
    static PointOrigin corner(std::size_t i) { return {Kind::Corner, i, i}; }
    static PointOrigin root(std::size_t i, std::size_t j) {
        return {Kind::EdgeRoot, std::min(i, j), std::max(i, j)};
    }
    bool known() const { return kind != Kind::Unknown; }
    bool operator==(const PointOrigin&) const = default;
};

struct OriginalCornerData {
    Point point{};
    Real phi{0.0};
    Real isovalue{0.0};
    Real signed_band{0.0};
    Real actual_signed{0.0};
    bool canonicalization_changed{false};
    bool available{false};
};

struct StrictConstructionObservation {
    LinearCornerStrictBranch state{LinearCornerStrictBranch::Unchecked};
    void unresolved() { state = LinearCornerStrictBranch::ModifiedOrUnresolved; }
    void combine(LinearCornerStrictBranch source) {
        if (source == LinearCornerStrictBranch::ModifiedOrUnresolved) unresolved();
    }
};

bool structuralRepeat(const Point& a, PointOrigin ao, const Point& b, PointOrigin bo)
{
    return ao.known() && ao == bo && a == b &&
        std::all_of(a.begin(), a.end(), [](Real x) { return std::isfinite(x); });
}

[[nodiscard]] detail::OriginRelation originRelation(const PointOrigin& a,
                                                    const PointOrigin& b) noexcept
{
    if (!a.known() || !b.known()) {
        return detail::OriginRelation::Unknown;
    }
    return a == b ? detail::OriginRelation::SameOriginal
                  : detail::OriginRelation::DistinctOriginal;
}

void observeAssessedRepeat(StrictConstructionObservation& observation,
                           const Point& a,
                           PointOrigin ao,
                           const detail::PointAssessment& aa,
                           const Point& b,
                           PointOrigin bo,
                           const detail::PointAssessment& ba,
                           Real tolerance,
                           detail::DistanceObservation distance)
{
    const auto assessment = detail::assessDistance(
        aa, a, ba, b, tolerance, 3u, originRelation(ao, bo), distance);
    if (!assessment.available()) {
        observation.unresolved();
    }
}

struct SignedPoint {
    Point point{{0.0, 0.0, 0.0}};
    Real value{0.0};
    PointOrigin origin{};
    OriginalCornerData original{};
    detail::PointAssessment assessment{};
};

[[nodiscard]] detail::PointAssessment assessCrossing(
    const SignedPoint& a,
    const SignedPoint& b,
    const Point& emitted,
    Real denominator,
    Real quotient,
    Real clamped,
    bool division_taken) noexcept
{
    if (a.value == Real{0.0} && b.value != Real{0.0}) {
        return a.assessment;
    }
    if (b.value == Real{0.0} && a.value != Real{0.0}) {
        return b.assessment;
    }
    if (!a.original.available || !b.original.available ||
        a.origin.kind != PointOrigin::Kind::Corner ||
        b.origin.kind != PointOrigin::Kind::Corner) {
        return {};
    }
    detail::OriginalEdgeObservation input;
    input.a = a.original.point;
    input.b = b.original.point;
    input.emitted = emitted;
    input.phi_a = a.original.phi;
    input.phi_b = b.original.phi;
    input.isovalue = a.original.isovalue;
    input.signed_band = a.original.signed_band;
    input.actual_signed_a = a.original.actual_signed;
    input.actual_signed_b = b.original.actual_signed;
    input.actual_denominator = denominator;
    input.actual_quotient = quotient;
    input.actual_clamped = clamped;
    input.canonicalization_changed =
        a.original.canonicalization_changed ||
        b.original.canonicalization_changed ||
        a.original.isovalue != b.original.isovalue ||
        a.original.signed_band != b.original.signed_band;
    input.helper_denominator_guard = true;
    input.division_taken = division_taken;
    return detail::assessOriginalEdge(input);
}

[[nodiscard]] Point add(const Point& a, const Point& b) noexcept
{
    return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
}

[[nodiscard]] Point sub(const Point& a, const Point& b) noexcept
{
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

[[nodiscard]] Point scale(const Point& a, Real value) noexcept
{
    return {{value * a[0], value * a[1], value * a[2]}};
}

[[nodiscard]] Real dot(const Point& a, const Point& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] Point cross(const Point& a, const Point& b) noexcept
{
    return {{a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]}};
}

[[nodiscard]] Real norm(const Point& point) noexcept
{
    return std::sqrt(dot(point, point));
}

[[nodiscard]] Point unit(const Point& point, const Point& fallback) noexcept
{
    const Real length = norm(point);
    return length > Real{1.0e-30}
               ? scale(point, Real{1.0} / length)
               : fallback;
}

[[nodiscard]] Point interpolate(const Point& a, const Point& b, Real t) noexcept
{
    return add(scale(a, Real{1.0} - t), scale(b, t));
}

[[nodiscard]] bool finitePoint(const Point& point) noexcept
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) &&
           std::isfinite(point[2]);
}

[[nodiscard]] bool samePoint(const Point& a,
                             const Point& b,
                             Real tolerance) noexcept
{
    return norm(sub(a, b)) <= tolerance;
}

[[nodiscard]] std::vector<std::size_t> localFaceCorners(ElementType type,
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

[[nodiscard]] std::vector<Point> referenceCellNodes(ElementType type,
                                                     std::size_t count)
{
    std::vector<Point> points;
    points.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto xi = basis::ReferenceNodeLayout::get_node_coords(type, i);
        points.push_back({{xi[0], xi[1], xi[2]}});
    }
    return points;
}

[[nodiscard]] Point centroid(const std::vector<Point>& points) noexcept
{
    Point value{{0.0, 0.0, 0.0}};
    for (const auto& point : points) {
        value = add(value, point);
    }
    return points.empty()
               ? value
               : scale(value, Real{1.0} / static_cast<Real>(points.size()));
}

[[nodiscard]] Point outwardReferenceNormal(const std::vector<Point>& cell,
                                           const std::vector<Point>& face,
                                           int dimension)
{
    if (dimension == 2) {
        if (face.size() != 2u) {
            throw std::invalid_argument(
                "generated active boundary requires a two-node reference edge");
        }
        const auto edge = sub(face[1], face[0]);
        Point normal{{edge[1], -edge[0], 0.0}};
        const auto outward = sub(centroid(face), centroid(cell));
        if (dot(normal, outward) < Real{0.0}) {
            normal = scale(normal, Real{-1.0});
        }
        return unit(normal, {{1.0, 0.0, 0.0}});
    }
    if (dimension == 3 && face.size() >= 3u) {
        Point normal = cross(sub(face[1], face[0]), sub(face[2], face[0]));
        const auto outward = sub(centroid(face), centroid(cell));
        if (dot(normal, outward) < Real{0.0}) {
            normal = scale(normal, Real{-1.0});
        }
        return unit(normal, {{1.0, 0.0, 0.0}});
    }
    throw std::invalid_argument(
        "generated active boundary requires a 2D or 3D parent cell");
}

[[nodiscard]] Real polygonMeasure(const std::vector<Point>& polygon,
                                  int dimension) noexcept
{
    if (dimension == 2) {
        return polygon.size() == 2u ? norm(sub(polygon[1], polygon[0]))
                                    : Real{0.0};
    }
    if (polygon.size() < 3u) {
        return Real{0.0};
    }
    const auto& origin = polygon.front();
    Real measure{0.0};
    for (std::size_t i = 1u; i + 1u < polygon.size(); ++i) {
        measure += Real{0.5} *
                   norm(cross(sub(polygon[i], origin),
                              sub(polygon[i + 1u], origin)));
    }
    return measure;
}

[[nodiscard]] bool isActive(Real value,
                            geometry::CutIntegrationSide side) noexcept
{
    return side == geometry::CutIntegrationSide::Negative
               ? value <= Real{0.0}
               : value >= Real{0.0};
}

[[nodiscard]] SignedPoint crossing(const SignedPoint& a,
                                   const SignedPoint& b,
                                   StrictConstructionObservation& observation) noexcept
{
    const Real denominator = a.value - b.value;
    const bool division_taken =
        std::abs(denominator) > Real{1.0e-30};
    const Real quotient =
        division_taken ? a.value / denominator : Real{0.0};
    const Real t = division_taken
                       ? std::clamp(quotient, Real{0.0}, Real{1.0})
                       : Real{0.5};
    PointOrigin origin;
    if (a.value == 0 && b.value != 0) origin = a.origin;
    else if (b.value == 0 && a.value != 0) origin = b.origin;
    else if (a.origin.kind == PointOrigin::Kind::Corner &&
             b.origin.kind == PointOrigin::Kind::Corner &&
             ((a.value < 0 && b.value > 0) || (a.value > 0 && b.value < 0))) {
        origin = PointOrigin::root(a.origin.first, b.origin.first);
    }
    if (!(std::abs(denominator) > Real{1.0e-30}) || !std::isfinite(denominator)) {
        observation.unresolved();
        origin = {};
    } else {
        if (!std::isfinite(quotient) || quotient < 0 || quotient > 1 ||
            (a.value != 0 && b.value != 0 &&
             !(quotient > 0 && quotient < 1))) {
            observation.unresolved();
        }
    }
    if (!origin.known()) observation.unresolved();
    const auto point = interpolate(a.point, b.point, t);
    return {point, Real{0.0}, origin, {},
            assessCrossing(a, b, point, denominator, quotient, t,
                           division_taken)};
}

[[nodiscard]] std::vector<SignedPoint> clipPolygon(
    const std::vector<SignedPoint>& polygon,
    geometry::CutIntegrationSide side,
    StrictConstructionObservation& observation)
{
    std::vector<SignedPoint> clipped;
    if (polygon.empty()) {
        return clipped;
    }
    SignedPoint previous = polygon.back();
    bool previous_inside = isActive(previous.value, side);
    for (const auto& current : polygon) {
        const bool current_inside = isActive(current.value, side);
        if (previous_inside && current_inside) {
            clipped.push_back(current);
        } else if (previous_inside && !current_inside) {
            clipped.push_back(crossing(previous, current, observation));
        } else if (!previous_inside && current_inside) {
            clipped.push_back(crossing(previous, current, observation));
            clipped.push_back(current);
        }
        previous = current;
        previous_inside = current_inside;
    }
    return clipped;
}

void removeDuplicatePolygonVertices(std::vector<Point>& points,
                                    std::vector<PointOrigin>& origins,
                                    std::vector<detail::PointAssessment>& assessments,
                                    Real tolerance,
                                    StrictConstructionObservation& observation)
{
    std::vector<Point> unique;
    std::vector<PointOrigin> unique_origins;
    std::vector<detail::PointAssessment> unique_assessments;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto& point = points[i];
        if (unique.empty()) {
            unique.push_back(point);
            unique_origins.push_back(origins[i]);
            unique_assessments.push_back(assessments[i]);
            continue;
        }
        const Real distance = norm(sub(unique.back(), point));
        const bool same = distance <= tolerance;
        observeAssessedRepeat(
            observation, unique.back(), unique_origins.back(),
            unique_assessments.back(), point, origins[i], assessments[i],
            tolerance,
            detail::DistanceObservation{true, distance, same, same});
        if (!same) {
            unique.push_back(point);
            unique_origins.push_back(origins[i]);
            unique_assessments.push_back(assessments[i]);
        }
    }
    if (unique.size() > 1u) {
        const Real distance = norm(sub(unique.front(), unique.back()));
        const bool same = distance <= tolerance;
        observeAssessedRepeat(
            observation, unique.front(), unique_origins.front(),
            unique_assessments.front(), unique.back(), unique_origins.back(),
            unique_assessments.back(), tolerance,
            detail::DistanceObservation{true, distance, same, same});
        if (same) {
            unique.pop_back();
            unique_origins.pop_back();
            unique_assessments.pop_back();
        }
    }
    for (std::size_t i = 0; i < unique.size(); ++i) {
        for (std::size_t j = i + 1u; j < unique.size(); ++j) {
            observeAssessedRepeat(
                observation, unique[i], unique_origins[i],
                unique_assessments[i], unique[j], unique_origins[j],
                unique_assessments[j], tolerance,
                detail::DistanceObservation{});
        }
    }
    points = std::move(unique);
    origins = std::move(unique_origins);
    assessments = std::move(unique_assessments);
}

[[nodiscard]] std::vector<geometry::CutQuadraturePoint> segmentQuadrature(
    const Point& a,
    const Point& b,
    const Point& normal,
    int order)
{
    const Real measure = norm(sub(b, a));
    std::vector<geometry::CutQuadraturePoint> points;
    const auto append = [&](Real t, Real fraction) {
        const auto point = interpolate(a, b, t);
        points.push_back(geometry::CutQuadraturePoint{
            .point = point,
            .normal = normal,
            .weight = measure * fraction,
            .parent_coordinate = point,
            .reference_measure_factor = measure});
    };
    if (order <= 1) {
        append(Real{0.5}, Real{1.0});
    } else if (order <= 3) {
        constexpr Real offset = Real{0.28867513459481288225};
        append(Real{0.5} - offset, Real{0.5});
        append(Real{0.5} + offset, Real{0.5});
    } else {
        constexpr Real offset = Real{0.38729833462074168852};
        append(Real{0.5} - offset, Real{5.0} / Real{18.0});
        append(Real{0.5}, Real{4.0} / Real{9.0});
        append(Real{0.5} + offset, Real{5.0} / Real{18.0});
    }
    return points;
}

[[nodiscard]] std::vector<geometry::CutQuadraturePoint> polygonQuadrature(
    const std::vector<Point>& polygon,
    const Point& normal,
    int order,
    const std::vector<PointOrigin>& origins,
    StrictConstructionObservation& observation)
{
    std::vector<geometry::CutQuadraturePoint> points;
    if (polygon.size() < 3u) {
        return points;
    }
    const auto& origin = polygon.front();
    for (std::size_t i = 1u; i + 1u < polygon.size(); ++i) {
        const auto& b = polygon[i];
        const auto& c = polygon[i + 1u];
        const Real area = Real{0.5} *
                          norm(cross(sub(b, origin), sub(c, origin)));
        if (!(area > Real{0.0})) {
            const bool structural_zero =
                structuralRepeat(origin, origins.front(), b, origins[i]) ||
                structuralRepeat(origin, origins.front(), c, origins[i + 1u]) ||
                structuralRepeat(b, origins[i], c, origins[i + 1u]);
            if (!(area == 0 && structural_zero)) observation.unresolved();
            continue;
        }
        const auto append = [&](Real l0, Real l1, Real l2, Real fraction) {
            const auto point = add(add(scale(origin, l0), scale(b, l1)),
                                   scale(c, l2));
            points.push_back(geometry::CutQuadraturePoint{
                .point = point,
                .normal = normal,
                .weight = area * fraction,
                .parent_coordinate = point,
                .reference_measure_factor = area});
        };
        if (order <= 1) {
            constexpr Real third = Real{1.0} / Real{3.0};
            append(third, third, third, Real{1.0});
        } else {
            constexpr Real high = Real{2.0} / Real{3.0};
            constexpr Real low = Real{1.0} / Real{6.0};
            constexpr Real weight = Real{1.0} / Real{3.0};
            append(high, low, low, weight);
            append(low, high, low, weight);
            append(low, low, high, weight);
        }
    }
    return points;
}

[[nodiscard]] std::uint64_t stableFragmentId(
    const GeneratedActiveBoundaryRequest& request,
    GlobalIndex parent_cell_identity,
    GlobalIndex parent_face_identity) noexcept
{
    std::uint64_t hash = 1469598103934665603ull;
    const auto mix = [&hash](std::uint64_t value) noexcept {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    mix(static_cast<std::uint64_t>(request.resolvedActiveBoundaryMarker()));
    mix(static_cast<std::uint64_t>(parent_cell_identity));
    mix(static_cast<std::uint64_t>(parent_face_identity));
    mix(static_cast<std::uint64_t>(request.side));
    mix(request.source_value_revision);
    mix(request.mesh_geometry_revision);
    mix(request.mesh_topology_revision);
    mix(request.ownership_revision);
    return hash;
}

[[nodiscard]] std::vector<const GeneratedInterfaceBoundaryIntersectionFragment*>
contactFragmentsForFace(
    const GeneratedInterfaceBoundaryIntersectionDomain& contact_domain,
    MeshIndex cell,
    MeshIndex face)
{
    std::vector<const GeneratedInterfaceBoundaryIntersectionFragment*> out;
    for (const auto& fragment : contact_domain.fragments()) {
        if (fragment.active() && fragment.parent_cell == cell &&
            fragment.parent_face == face) {
            out.push_back(&fragment);
        }
    }
    return out;
}

[[nodiscard]] bool matchesAuthoritativeContactVertex(
    const Point& point,
    const std::vector<const GeneratedInterfaceBoundaryIntersectionFragment*>&
        contacts,
    Real tolerance) noexcept
{
    for (const auto* contact : contacts) {
        for (const auto& vertex : contact->vertices) {
            if (samePoint(point, vertex, tolerance)) {
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] bool allZero(const std::vector<SignedPoint>& points) noexcept
{
    return std::all_of(points.begin(), points.end(), [](const auto& point) {
        return point.value == Real{0.0};
    });
}

[[nodiscard]] bool hasAuthoritativeInterfaceFragment(
    const LevelSetInterfaceDomain& domain,
    MeshIndex cell) noexcept
{
    return std::any_of(
        domain.fragments().begin(),
        domain.fragments().end(),
        [cell](const auto& fragment) {
            return fragment.active() && fragment.parent_cell == cell;
        });
}

[[nodiscard]] std::optional<geometry::CutIntegrationSide>
authoritativeFullCellSide(const LevelSetInterfaceDomain& domain,
                          MeshIndex cell)
{
    std::optional<geometry::CutIntegrationSide> side;
    for (const auto& region : domain.volumeRegions()) {
        if (!region.active() || region.parent_cell != cell ||
            !region.full_cell_equivalent) {
            continue;
        }
        if (side.has_value() && *side != region.side) {
            throw std::invalid_argument(
                "sharp active-boundary construction found conflicting authoritative full-cell phases");
        }
        side = region.side;
    }
    return side;
}

[[nodiscard]] bool sameRevisionKey(
    const GeneratedActiveBoundaryRequest& active,
    const CutInterfaceDomainRequest& interface,
    const GeneratedInterfaceBoundaryIntersectionRequest& contact) noexcept
{
    return active.source.identifier() == interface.source.identifier() &&
           active.source.identifier() == contact.source.identifier() &&
           active.source.layout_revision == interface.source.layout_revision &&
           active.source.layout_revision == contact.source.layout_revision &&
           active.source_value_revision == interface.source.value_revision &&
           active.source_value_revision == contact.source_value_revision &&
           active.mesh_geometry_revision == interface.mesh_geometry_revision &&
           active.mesh_geometry_revision == contact.mesh_geometry_revision &&
           active.mesh_topology_revision == interface.mesh_topology_revision &&
           active.mesh_topology_revision == contact.mesh_topology_revision &&
           active.ownership_revision == interface.ownership_revision &&
           active.ownership_revision == contact.ownership_revision &&
           active.quadrature_policy_key == interface.quadrature_policy_key &&
           active.quadrature_policy_key == contact.quadrature_policy_key;
}

struct RepresentedImplicitProvenance {
    std::string geometry_mode{};
    std::string backend{};
    std::string fallback_status{};
};

[[nodiscard]] RepresentedImplicitProvenance representedImplicitForCell(
    const LevelSetInterfaceDomain& domain,
    MeshIndex cell)
{
    RepresentedImplicitProvenance represented;
    represented.geometry_mode = domain.request().implicit_geometry_mode;
    const auto merge = [&](const std::string& backend,
                           const std::string& fallback_status) {
        const std::string selected_backend =
            backend.empty() ? domain.request().implicit_quadrature_backend
                            : backend;
        const std::string selected_fallback =
            fallback_status.empty()
                ? domain.request().implicit_fallback_status
                : fallback_status;
        if (!represented.backend.empty() &&
            represented.backend != selected_backend) {
            throw std::invalid_argument(
                "sharp active-boundary construction found multiple represented implicit backends on one parent cell");
        }
        if (!represented.fallback_status.empty() &&
            represented.fallback_status != selected_fallback) {
            throw std::invalid_argument(
                "sharp active-boundary construction found multiple implicit fallback states on one parent cell");
        }
        represented.backend = selected_backend;
        represented.fallback_status = selected_fallback;
    };
    for (const auto& fragment : domain.fragments()) {
        if (fragment.active() && fragment.parent_cell == cell) {
            merge(fragment.implicit_quadrature_backend,
                  fragment.implicit_fallback_status);
        }
    }
    for (const auto& region : domain.volumeRegions()) {
        if (region.active() && region.parent_cell == cell) {
            merge(region.implicit_quadrature_backend,
                  region.implicit_fallback_status);
        }
    }
    if (represented.backend.empty()) {
        represented.backend = domain.request().implicit_quadrature_backend;
        represented.fallback_status =
            domain.request().implicit_fallback_status;
    }
    if (represented.geometry_mode.empty() || represented.backend.empty() ||
        represented.backend == "Auto" || represented.backend == "Unknown") {
        throw std::invalid_argument(
            "sharp active-boundary construction requires an explicit represented implicit backend");
    }
    // The current exterior-face clipping algorithm evaluates only corner
    // values and linearly interpolates roots.  Refuse other representations
    // until their boundary restriction is implemented, even on apparently
    // uncut faces where an interior sign change could otherwise be missed.
    if (represented.backend != "LinearCorner") {
        throw std::invalid_argument(
            "sharp active-boundary clipping currently supports only the LinearCorner represented implicit backend");
    }
    return represented;
}

} // namespace

std::string GeneratedActiveBoundaryMarkerKey::stableKey() const
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
    append_component(
        std::to_string(
            static_cast<int>(side)));
    return key;
}

std::uint64_t stableGeneratedActiveBoundaryMarkerHash(
    const GeneratedActiveBoundaryMarkerKey& key)
{
    std::uint64_t hash = 1469598103934665603ull;
    for (const char value : key.stableKey()) {
        hash ^= static_cast<unsigned char>(value);
        hash *= 1099511628211ull;
    }
    return hash;
}

int stableGeneratedActiveBoundaryMarker(
    const GeneratedActiveBoundaryMarkerKey& key,
    int marker_base,
    int marker_range)
{
    if (key.requested_marker >= 0) {
        return key.requested_marker;
    }
    if (marker_base < 0 || marker_range <= 0 ||
        key.boundary_marker < 0 || key.interface_marker < 0 ||
        key.side == geometry::CutIntegrationSide::Interface ||
        !key.source.valid()) {
        throw std::invalid_argument(
            "generated active-boundary marker key is invalid");
    }
    return marker_base + static_cast<int>(
                             stableGeneratedActiveBoundaryMarkerHash(key) %
                             static_cast<std::uint64_t>(marker_range));
}

bool GeneratedActiveBoundaryRequest::valid() const noexcept
{
    return source.valid() && !generated_domain_id.empty() &&
           interface_marker >= 0 && boundary_marker >= 0 &&
           side != geometry::CutIntegrationSide::Interface &&
           std::isfinite(isovalue) &&
           std::isfinite(tolerance) &&
           tolerance > Real{0.0} && quadrature_order >= 0 &&
           frame == geometry::CutGeometryFrame::Reference &&
           (source.value_revision == 0u ||
            source_value_revision == 0u ||
            source.value_revision == source_value_revision);
}

int GeneratedActiveBoundaryRequest::resolvedActiveBoundaryMarker() const
{
    GeneratedActiveBoundaryMarkerKey key;
    key.source = source;
    key.domain_id = generated_domain_id;
    key.isovalue = isovalue;
    key.interface_marker = interface_marker;
    key.boundary_marker = boundary_marker;
    key.side = side;
    key.requested_marker = active_boundary_marker;
    return stableGeneratedActiveBoundaryMarker(key);
}

bool GeneratedActiveBoundaryFragment::active() const noexcept
{
    return interface_marker >= 0 && boundary_marker >= 0 &&
           active_boundary_marker >= 0 && parent_cell >= 0 &&
           parent_face >= 0 && side != geometry::CutIntegrationSide::Interface &&
           !represented_implicit_geometry_mode.empty() &&
           !represented_implicit_quadrature_backend.empty() &&
           std::isfinite(measure) && measure > Real{0.0} &&
           !quadrature_points.empty();
}

geometry::CutQuadratureRule
GeneratedActiveBoundaryFragment::toCutQuadratureRule(
    const GeneratedActiveBoundaryRequest& request) const
{
    if (!active()) {
        throw std::invalid_argument(
            "cannot convert an inactive generated active-boundary fragment");
    }
    geometry::CutQuadratureRule rule;
    rule.kind = geometry::CutQuadratureKind::Interface;
    rule.side = side;
    rule.geometric_dimension = -1;
    rule.points = quadrature_points;
    rule.measure = measure;
    rule.parent_measure = parent_measure;
    rule.volume_fraction = parent_measure > Real{0.0}
                               ? measure / parent_measure
                               : Real{0.0};
    rule.exact_for_constants = true;
    rule.exact_polynomial_order = achieved_quadrature_order;
    rule.policy.kind = geometry::CutQuadratureConstructionKind::TopologySubdivision;
    rule.policy.polynomial_order = achieved_quadrature_order;
    rule.policy.moment_fitted = false;
    rule.policy.name = full_face_equivalent
                           ? "full-active-exterior-boundary"
                           : "sharp-clipped-active-exterior-boundary";
    rule.provenance.embedded_geometry_id = request.source.identifier();
    rule.provenance.cut_topology_id = topology_id;
    rule.provenance.parent_entity = parent_cell;
    rule.provenance.parent_boundary_entity = parent_face;
    rule.provenance.parent_entity_global_id = parent_cell_global_id;
    rule.provenance.parent_boundary_entity_global_id =
        parent_face_global_id;
    rule.provenance.owner_rank = owner_rank;
    rule.provenance.marker = active_boundary_marker;
    rule.provenance.cut_topology_revision = stable_id;
    rule.provenance.predicate_policy_key = request.quadrature_policy_key;
    rule.provenance.source_value_revision = request.source_value_revision;
    if (source_interface_stable_ids.size() == 1u) {
        rule.provenance.source_stable_id = source_interface_stable_ids.front();
    }
    rule.provenance.construction = rule.policy.kind;
    rule.provenance.frame = request.frame;
    rule.provenance.implicit_geometry_mode =
        represented_implicit_geometry_mode;
    rule.provenance.implicit_quadrature_backend =
        represented_implicit_quadrature_backend;
    rule.provenance.selected_implicit_quadrature_backend =
        represented_implicit_quadrature_backend;
    rule.provenance.implicit_fallback_status =
        represented_implicit_fallback_status;
    rule.provenance.requested_quadrature_order = request.quadrature_order;
    rule.provenance.achieved_quadrature_order = achieved_quadrature_order;
    rule.provenance_id = request.source.identifier();
    rule.frame = request.frame;
    rule.full_cell_equivalent = full_face_equivalent;
    return rule;
}

GeneratedActiveBoundaryDomain::GeneratedActiveBoundaryDomain(
    GeneratedActiveBoundaryRequest request)
    : request_(std::move(request))
{
    if (!request_.valid()) {
        throw std::invalid_argument(
            "generated active-boundary request is invalid");
    }
    request_.active_boundary_marker = request_.resolvedActiveBoundaryMarker();
}

const GeneratedActiveBoundaryRequest&
GeneratedActiveBoundaryDomain::request() const noexcept
{
    return request_;
}

int GeneratedActiveBoundaryDomain::marker() const noexcept
{
    return request_.active_boundary_marker;
}

bool GeneratedActiveBoundaryDomain::empty() const noexcept
{
    return fragments_.empty();
}

const std::vector<GeneratedActiveBoundaryFragment>&
GeneratedActiveBoundaryDomain::fragments() const noexcept
{
    return fragments_;
}

void GeneratedActiveBoundaryDomain::addFragment(
    GeneratedActiveBoundaryFragment fragment)
{
    if (fragment.interface_marker < 0) {
        fragment.interface_marker = request_.interface_marker;
    }
    if (fragment.boundary_marker < 0) {
        fragment.boundary_marker = request_.boundary_marker;
    }
    if (fragment.active_boundary_marker < 0) {
        fragment.active_boundary_marker = request_.active_boundary_marker;
    }
    if (fragment.side == geometry::CutIntegrationSide::Interface) {
        fragment.side = request_.side;
    }
    if (fragment.local_fragment_index == INVALID_LOCAL_INDEX) {
        fragment.local_fragment_index =
            static_cast<LocalIndex>(fragments_.size());
    }
    if (fragment.stable_id == 0u) {
        const auto cell_identity =
            fragment.parent_cell_global_id != INVALID_GLOBAL_INDEX
                ? fragment.parent_cell_global_id
                : static_cast<GlobalIndex>(fragment.parent_cell);
        const auto face_identity =
            fragment.parent_face_global_id != INVALID_GLOBAL_INDEX
                ? fragment.parent_face_global_id
                : static_cast<GlobalIndex>(fragment.parent_face);
        fragment.stable_id = stableFragmentId(
            request_, cell_identity, face_identity);
    }
    std::sort(fragment.source_contact_stable_ids.begin(),
              fragment.source_contact_stable_ids.end());
    fragment.source_contact_stable_ids.erase(
        std::unique(fragment.source_contact_stable_ids.begin(),
                    fragment.source_contact_stable_ids.end()),
        fragment.source_contact_stable_ids.end());
    std::sort(fragment.source_interface_stable_ids.begin(),
              fragment.source_interface_stable_ids.end());
    fragment.source_interface_stable_ids.erase(
        std::unique(fragment.source_interface_stable_ids.begin(),
                    fragment.source_interface_stable_ids.end()),
        fragment.source_interface_stable_ids.end());
    if (!fragment.active()) {
        throw std::invalid_argument(
            "generated active-boundary fragment is invalid");
    }
    fragments_.push_back(std::move(fragment));
}

GeneratedActiveBoundarySummary
GeneratedActiveBoundaryDomain::summary() const noexcept
{
    GeneratedActiveBoundarySummary summary;
    for (const auto& fragment : fragments_) {
        if (!fragment.active()) {
            continue;
        }
        ++summary.fragment_count;
        summary.quadrature_point_count += fragment.quadrature_points.size();
        summary.measure += fragment.measure;
        summary.parent_measure += fragment.parent_measure;
        if (fragment.full_face_equivalent) {
            ++summary.full_face_count;
        } else {
            ++summary.cut_face_count;
        }
    }
    return summary;
}

std::vector<geometry::CutQuadratureRule>
GeneratedActiveBoundaryDomain::boundaryQuadratureRules() const
{
    std::vector<geometry::CutQuadratureRule> rules;
    rules.reserve(fragments_.size());
    for (const auto& fragment : fragments_) {
        if (fragment.active()) {
            rules.push_back(fragment.toCutQuadratureRule(request_));
        }
    }
    std::sort(rules.begin(), rules.end(), cutQuadratureRuleDeterministicLess);
    return rules;
}

GeneratedActiveBoundaryDomain buildGeneratedActiveBoundaryDomain(
    GeneratedActiveBoundaryRequest request,
    const LevelSetInterfaceDomain& interface_domain,
    const GeneratedInterfaceBoundaryIntersectionDomain& contact_domain,
    const assembly::IMeshAccess& mesh,
    const GeneratedActiveBoundaryScalarField& scalar_field)
{
    if (!request.valid() || !scalar_field.valid()) {
        throw std::invalid_argument(
            "generated active-boundary construction requires a valid request and scalar field");
    }
    if (mesh.parallelSize() > 1 && !mesh.globalEntityIdsAvailable()) {
        throw std::invalid_argument(
            "distributed active-boundary geometry requires globally unique cell and face ids");
    }
    if (request.interface_marker != interface_domain.marker() ||
        request.boundary_marker != contact_domain.boundaryMarker() ||
        !sameRevisionKey(request,
                         interface_domain.request(),
                         contact_domain.request())) {
        throw std::invalid_argument(
            "generated active-boundary inputs do not share one geometry revision key");
    }
    GeneratedActiveBoundaryDomain domain(std::move(request));
    const auto& req = domain.request();

    mesh.forEachBoundaryFace(
        req.boundary_marker,
        [&](GlobalIndex face, GlobalIndex cell) {
            if (!mesh.isOwnedCell(cell)) {
                return;
            }
            const auto type = mesh.getCellType(cell);
            const auto represented = representedImplicitForCell(
                interface_domain, static_cast<MeshIndex>(cell));
            const auto local_face = mesh.getLocalFaceIndex(face, cell);
            const auto corners = localFaceCorners(type, local_face);
            if (corners.size() < 2u) {
                throw std::invalid_argument(
                    "sharp active-boundary clipping encountered an unsupported parent face");
            }
            std::vector<GlobalIndex> cell_nodes;
            mesh.getCellNodes(cell, cell_nodes);
            if (cell_nodes.empty()) {
                throw std::invalid_argument(
                    "sharp active-boundary clipping found an empty cell-node list");
            }
            const auto reference_nodes = referenceCellNodes(type, cell_nodes.size());
            std::vector<Point> face_points;
            StrictConstructionObservation observation;
            for (const auto& source : interface_domain.fragments()) {
                if (source.parent_cell == static_cast<MeshIndex>(cell))
                    observation.combine(source.construction_observation);
            }
            for (const auto& source : interface_domain.volumeRegions()) {
                if (source.parent_cell == static_cast<MeshIndex>(cell))
                    observation.combine(source.construction_observation);
            }
            std::vector<SignedPoint> signed_face;
            face_points.reserve(corners.size());
            signed_face.reserve(corners.size());
            for (const auto corner : corners) {
                if (corner >= cell_nodes.size() || corner >= reference_nodes.size()) {
                    throw std::invalid_argument(
                        "sharp active-boundary clipping found incomplete face connectivity");
                }
                const Real original_value =
                    scalar_field.value_at_node(cell_nodes[corner]);
                const Real actual_signed = original_value - req.isovalue;
                Real value = actual_signed;
                if (!std::isfinite(value)) {
                    throw std::invalid_argument(
                        "sharp active-boundary clipping found a non-finite level-set value");
                }
                if (std::abs(value) <= req.tolerance) {
                    observation.unresolved();
                    value = Real{0.0};
                }
                face_points.push_back(reference_nodes[corner]);
                signed_face.push_back({
                    reference_nodes[corner], value, PointOrigin::corner(corner),
                    OriginalCornerData{
                        reference_nodes[corner], original_value, req.isovalue,
                        req.tolerance, actual_signed, actual_signed != value,
                        true},
                    detail::assessOriginalCorner(reference_nodes[corner],
                                                 reference_nodes[corner])});
            }
            if (!hasAuthoritativeInterfaceFragment(
                    interface_domain,
                    static_cast<MeshIndex>(cell))) {
                const auto full_side = authoritativeFullCellSide(
                    interface_domain,
                    static_cast<MeshIndex>(cell));
                if (!full_side.has_value()) {
                    throw std::invalid_argument(
                        "sharp active-boundary construction found neither an authoritative interface fragment nor a full-cell phase");
                }
                // A LinearCorner cell whose surface fragment falls below the
                // source geometry's measure tolerance is represented as one
                // full dominant-phase volume.  Its exterior trace must inherit
                // that same represented phase instead of reconstructing a
                // smaller scalar-field cut that has no authoritative contact.
                const Real represented_value =
                    *full_side == geometry::CutIntegrationSide::Negative
                        ? Real{-1.0}
                        : Real{1.0};
                const bool original_negative = std::any_of(signed_face.begin(), signed_face.end(),
                    [](const auto& point) { return point.value < 0; });
                const bool original_positive = std::any_of(signed_face.begin(), signed_face.end(),
                    [](const auto& point) { return point.value > 0; });
                if (original_negative && original_positive) observation.unresolved();
                for (auto& point : signed_face) {
                    point.value = represented_value;
                }
            }
            if (allZero(signed_face)) {
                throw std::invalid_argument(
                    "sharp active-boundary clipping encountered an unresolved face-aligned zero set");
            }
            const auto normal = outwardReferenceNormal(
                reference_nodes, face_points, mesh.dimension());
            const Real parent_measure =
                polygonMeasure(face_points, mesh.dimension());
            if (!(parent_measure > Real{0.0}) || !std::isfinite(parent_measure)) {
                throw std::invalid_argument(
                    "sharp active-boundary clipping found a degenerate parent face");
            }

            const auto clipped_signed = clipPolygon(signed_face, req.side, observation);
            std::vector<Point> clipped;
            std::vector<PointOrigin> origins;
            std::vector<detail::PointAssessment> assessments;
            clipped.reserve(clipped_signed.size());
            origins.reserve(clipped_signed.size());
            assessments.reserve(clipped_signed.size());
            for (const auto& point : clipped_signed) {
                clipped.push_back(point.point);
                origins.push_back(point.origin);
                assessments.push_back(point.assessment);
            }
            removeDuplicatePolygonVertices(clipped, origins, assessments,
                                           req.tolerance, observation);
            const std::size_t minimum_points = mesh.dimension() == 2 ? 2u : 3u;
            if (clipped.size() < minimum_points) {
                observation.unresolved();
                return;
            }
            const Real measure = polygonMeasure(clipped, mesh.dimension());
            if (!(measure > req.tolerance * req.tolerance) ||
                !std::isfinite(measure)) {
                observation.unresolved();
                return;
            }

            const auto contacts = contactFragmentsForFace(
                contact_domain,
                static_cast<MeshIndex>(cell),
                static_cast<MeshIndex>(face));
            bool cut_face = false;
            for (const auto& point : clipped_signed) {
                if (point.value != Real{0.0}) {
                    continue;
                }
                const bool is_original_zero = std::any_of(
                    signed_face.begin(), signed_face.end(), [&](const auto& original) {
                        return original.value == Real{0.0} &&
                               samePoint(original.point, point.point, req.tolerance);
                    });
                const bool separates_signs = std::any_of(
                    signed_face.begin(), signed_face.end(), [](const auto& original) {
                        return original.value < Real{0.0};
                    }) &&
                    std::any_of(signed_face.begin(), signed_face.end(), [](const auto& original) {
                        return original.value > Real{0.0};
                    });
                if (!is_original_zero || separates_signs) {
                    cut_face = true;
                    if (!matchesAuthoritativeContactVertex(
                            point.point,
                            contacts,
                            Real{128.0} * req.tolerance)) {
                        throw std::invalid_argument(
                            "sharp active-boundary clipping produced a root that is not an authoritative contact-trace vertex");
                    }
                }
            }
            if (cut_face && contacts.empty()) {
                throw std::invalid_argument(
                    "sharp active-boundary clipping found a cut face without an authoritative contact fragment");
            }

            GeneratedActiveBoundaryFragment fragment;
            fragment.parent_cell = cell;
            fragment.parent_face = face;
            fragment.parent_cell_global_id = mesh.globalEntityIdsAvailable()
                                                 ? mesh.getCellGlobalId(cell)
                                                 : cell;
            fragment.parent_face_global_id = mesh.globalEntityIdsAvailable()
                                                 ? mesh.getBoundaryFaceGlobalId(face)
                                                 : face;
            fragment.owner_rank =
                mesh.getBoundaryFaceOwnerRank(face, cell);
            fragment.side = req.side;
            fragment.represented_implicit_geometry_mode =
                represented.geometry_mode;
            fragment.represented_implicit_quadrature_backend =
                represented.backend;
            fragment.represented_implicit_fallback_status =
                represented.fallback_status;
            fragment.boundary_normal = normal;
            fragment.measure = measure;
            fragment.parent_measure = parent_measure;
            fragment.full_face_equivalent =
                std::abs(measure - parent_measure) <=
                Real{128.0} * std::numeric_limits<Real>::epsilon() *
                    std::max(Real{1.0}, parent_measure);
            fragment.achieved_quadrature_order =
                mesh.dimension() == 2
                    ? std::min(req.quadrature_order, 5)
                    : std::min(req.quadrature_order, 2);
            fragment.topology_id =
                "active-boundary:" +
                std::to_string(fragment.parent_face_global_id) + ":" +
                std::to_string(static_cast<int>(req.side));
            fragment.vertices = clipped;
            for (const auto* contact : contacts) {
                if ((contact->parent_cell_global_id != INVALID_GLOBAL_INDEX &&
                     contact->parent_cell_global_id !=
                         fragment.parent_cell_global_id) ||
                    (contact->parent_face_global_id != INVALID_GLOBAL_INDEX &&
                     contact->parent_face_global_id !=
                         fragment.parent_face_global_id) ||
                    (contact->owner_rank >= 0 &&
                     contact->owner_rank != fragment.owner_rank)) {
                    throw std::invalid_argument(
                        "sharp active-boundary fragment and contact trace have inconsistent global ownership metadata");
                }
                if (contact->represented_implicit_geometry_mode !=
                        fragment.represented_implicit_geometry_mode ||
                    contact->represented_implicit_quadrature_backend !=
                        fragment.represented_implicit_quadrature_backend ||
                    contact->represented_implicit_fallback_status !=
                        fragment.represented_implicit_fallback_status) {
                    throw std::invalid_argument(
                        "sharp active-boundary fragment and authoritative contact trace use different represented implicit backends");
                }
                fragment.source_contact_stable_ids.push_back(contact->stable_id);
                fragment.source_interface_stable_ids.push_back(
                    contact->source_interface_stable_id);
            }
            if (mesh.dimension() == 2) {
                fragment.quadrature_points = segmentQuadrature(
                    clipped[0],
                    clipped[1],
                    normal,
                    fragment.achieved_quadrature_order);
            } else {
                fragment.quadrature_points = polygonQuadrature(
                    clipped,
                    normal,
                    fragment.achieved_quadrature_order,
                    origins,
                    observation);
            }
            fragment.construction_observation = observation.state;
            domain.addFragment(std::move(fragment));
        });
    return domain;
}

GeneratedActiveBoundaryPartitionSummary
validateGeneratedActiveBoundaryPartition(
    const GeneratedActiveBoundaryDomain& negative_domain,
    const GeneratedActiveBoundaryDomain& positive_domain,
    const LevelSetInterfaceDomain& interface_domain,
    const GeneratedInterfaceBoundaryIntersectionDomain& contact_domain,
    const assembly::IMeshAccess& mesh)
{
    GeneratedActiveBoundaryPartitionSummary summary;
    const auto& negative = negative_domain.request();
    const auto& positive = positive_domain.request();
    if (negative.side != geometry::CutIntegrationSide::Negative ||
        positive.side != geometry::CutIntegrationSide::Positive ||
        negative.boundary_marker != positive.boundary_marker ||
        negative.interface_marker != positive.interface_marker ||
        negative.source_value_revision != positive.source_value_revision ||
        !sameRevisionKey(negative,
                         interface_domain.request(),
                         contact_domain.request()) ||
        !sameRevisionKey(positive,
                         interface_domain.request(),
                         contact_domain.request())) {
        ++summary.stale_revision_count;
        throw std::invalid_argument(
            "generated active-boundary phase domains do not share one revision key");
    }

    std::unordered_set<std::uint64_t> contact_ids;
    std::unordered_map<
        std::uint64_t,
        const GeneratedInterfaceBoundaryIntersectionFragment*>
        contact_by_id;
    for (const auto& contact : contact_domain.fragments()) {
        if (contact.active()) {
            contact_ids.insert(contact.stable_id);
            contact_by_id.emplace(contact.stable_id, &contact);
        }
    }
    summary.source_contact_fragment_count = contact_ids.size();
    std::unordered_set<std::uint64_t> referenced_contacts;
    std::map<GlobalIndex, std::array<Real, 3>> face_measures;
    const auto collect = [&](const GeneratedActiveBoundaryDomain& domain,
                             std::size_t side_index) {
        for (const auto& fragment : domain.fragments()) {
            if (!fragment.active()) {
                throw std::invalid_argument(
                    "generated active-boundary validator found an inactive retained fragment");
            }
            const auto local_cell =
                static_cast<GlobalIndex>(fragment.parent_cell);
            const auto local_face =
                static_cast<GlobalIndex>(fragment.parent_face);
            const auto expected_cell_id = mesh.globalEntityIdsAvailable()
                                              ? mesh.getCellGlobalId(local_cell)
                                              : local_cell;
            const auto expected_face_id = mesh.globalEntityIdsAvailable()
                                              ? mesh.getBoundaryFaceGlobalId(local_face)
                                              : local_face;
            const auto expected_owner =
                mesh.getBoundaryFaceOwnerRank(local_face, local_cell);
            if (fragment.parent_cell_global_id != expected_cell_id ||
                fragment.parent_face_global_id != expected_face_id ||
                fragment.owner_rank != expected_owner) {
                throw std::invalid_argument(
                    "generated active-boundary validator rejected stale global identity or ownership metadata");
            }
            const auto face_identity =
                fragment.parent_face_global_id != INVALID_GLOBAL_INDEX
                    ? fragment.parent_face_global_id
                    : static_cast<GlobalIndex>(fragment.parent_face);
            auto& measures = face_measures[face_identity];
            measures[side_index] += fragment.measure;
            measures[2] = fragment.parent_measure;
            if (!fragment.full_face_equivalent) {
                ++summary.cut_boundary_face_count;
            }
            for (const auto id : fragment.source_contact_stable_ids) {
                const auto contact = contact_by_id.find(id);
                if (contact == contact_by_id.end()) {
                    ++summary.orphan_source_reference_count;
                } else {
                    if (contact->second->parent_cell_global_id !=
                            fragment.parent_cell_global_id ||
                        contact->second->parent_face_global_id !=
                            fragment.parent_face_global_id ||
                        contact->second->owner_rank != fragment.owner_rank) {
                        throw std::invalid_argument(
                            "generated active-boundary source contact has inconsistent global identity or ownership");
                    }
                    referenced_contacts.insert(id);
                }
            }
            Real weight_sum{0.0};
            for (const auto& point : fragment.quadrature_points) {
                if (!finitePoint(point.point) ||
                    !finitePoint(point.parent_coordinate) ||
                    !std::isfinite(point.weight) ||
                    !(point.weight > Real{0.0}) ||
                    std::abs(norm(point.normal) - Real{1.0}) > Real{1.0e-10}) {
                    throw std::invalid_argument(
                        "generated active-boundary validator found an invalid quadrature point");
                }
                weight_sum += point.weight;
            }
            const Real tolerance = Real{256.0} *
                                   std::numeric_limits<Real>::epsilon() *
                                   std::max(Real{1.0}, fragment.measure);
            if (std::abs(weight_sum - fragment.measure) > tolerance) {
                throw std::invalid_argument(
                    "generated active-boundary quadrature does not integrate constants exactly");
            }
        }
    };
    collect(negative_domain, 0u);
    collect(positive_domain, 1u);
    summary.referenced_contact_fragment_count = referenced_contacts.size();
    if (summary.orphan_source_reference_count != 0u) {
        throw std::invalid_argument(
            "generated active-boundary geometry contains an orphan contact reference");
    }

    mesh.forEachBoundaryFace(
        negative.boundary_marker,
        [&](GlobalIndex face, GlobalIndex cell) {
            if (!mesh.isOwnedCell(cell)) {
                return;
            }
            ++summary.boundary_face_count;
            const auto face_identity = mesh.globalEntityIdsAvailable()
                                           ? mesh.getBoundaryFaceGlobalId(face)
                                           : face;
            const auto found = face_measures.find(face_identity);
            if (found == face_measures.end()) {
                throw std::invalid_argument(
                    "generated active-boundary phase partition omitted a physical boundary face");
            }
            const auto& measures = found->second;
            const Real error =
                std::abs(measures[0] + measures[1] - measures[2]);
            summary.max_partition_error =
                std::max(summary.max_partition_error, error);
            summary.total_boundary_measure += measures[2];
            summary.negative_boundary_measure += measures[0];
            summary.positive_boundary_measure += measures[1];
            const Real tolerance = Real{512.0} *
                                   std::numeric_limits<Real>::epsilon() *
                                   std::max(Real{1.0}, measures[2]);
            if (error > tolerance) {
                throw std::invalid_argument(
                    "generated active-boundary wet/dry measures do not partition the parent face");
            }
        });
    return summary;
}

} // namespace svmp::FE::interfaces
