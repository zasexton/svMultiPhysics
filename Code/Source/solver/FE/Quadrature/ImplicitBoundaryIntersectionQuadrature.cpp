/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Quadrature/ImplicitBoundaryIntersectionQuadrature.h"

#include "Basis/NodeOrderingConventions.h"
#include "Elements/ReferenceElement.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace svmp {
namespace FE {
namespace quadrature {
namespace {

constexpr Real kTiny = Real{1.0e-30};

struct SubentityVertex {
    LocalIndex local_node{INVALID_LOCAL_INDEX};
    std::array<Real, 3> reference{{0.0, 0.0, 0.0}};
    std::array<Real, 3> physical{{0.0, 0.0, 0.0}};
    Real signed_value{0.0};
};

struct CutCandidate {
    std::array<Real, 3> reference{{0.0, 0.0, 0.0}};
    std::array<Real, 3> physical{{0.0, 0.0, 0.0}};
    std::size_t edge_index{0};
    bool at_vertex{false};
};

[[nodiscard]] Real dot(const std::array<Real, 3>& a,
                       const std::array<Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] std::array<Real, 3> add(const std::array<Real, 3>& a,
                                      const std::array<Real, 3>& b) noexcept
{
    return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
}

[[nodiscard]] std::array<Real, 3> sub(const std::array<Real, 3>& a,
                                      const std::array<Real, 3>& b) noexcept
{
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
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

[[nodiscard]] std::array<Real, 3> interpolate(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b,
    Real t) noexcept
{
    return add(scale(a, Real{1.0} - t), scale(b, t));
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

void addUniqueCandidate(std::vector<CutCandidate>& candidates,
                        CutCandidate candidate,
                        Real tolerance)
{
    const auto duplicate =
        std::find_if(candidates.begin(), candidates.end(),
                     [&](const CutCandidate& existing) {
                         return samePoint(existing.reference,
                                          candidate.reference,
                                          tolerance) ||
                                samePoint(existing.physical,
                                          candidate.physical,
                                          tolerance);
                     });
    if (duplicate == candidates.end()) {
        candidates.push_back(std::move(candidate));
    }
}

[[nodiscard]] ElementType canonicalElement(ElementType type) noexcept
{
    switch (type) {
    case ElementType::Line3:
        return ElementType::Line2;
    case ElementType::Triangle6:
        return ElementType::Triangle3;
    case ElementType::Quad8:
    case ElementType::Quad9:
        return ElementType::Quad4;
    case ElementType::Tetra10:
        return ElementType::Tetra4;
    case ElementType::Hex20:
    case ElementType::Hex27:
        return ElementType::Hex8;
    case ElementType::Wedge15:
    case ElementType::Wedge18:
        return ElementType::Wedge6;
    case ElementType::Pyramid13:
    case ElementType::Pyramid14:
        return ElementType::Pyramid5;
    default:
        return type;
    }
}

[[nodiscard]] std::size_t cornerCount(ElementType type) noexcept
{
    switch (canonicalElement(type)) {
    case ElementType::Line2:
        return 2u;
    case ElementType::Triangle3:
        return 3u;
    case ElementType::Quad4:
        return 4u;
    case ElementType::Tetra4:
        return 4u;
    case ElementType::Hex8:
        return 8u;
    case ElementType::Wedge6:
        return 6u;
    case ElementType::Pyramid5:
        return 5u;
    default:
        return 0u;
    }
}

[[nodiscard]] std::array<Real, 3> referenceCoordinate(ElementType type,
                                                      std::size_t node)
{
    const auto base = canonicalElement(type);
    const auto xi =
        basis::ReferenceNodeLayout::get_node_coords(base, node);
    return {{xi[0], xi[1], xi[2]}};
}

[[nodiscard]] bool solve2x2(Real a00,
                            Real a01,
                            Real a10,
                            Real a11,
                            Real b0,
                            Real b1,
                            Real& x0,
                            Real& x1) noexcept
{
    const Real det = a00 * a11 - a01 * a10;
    if (!(std::abs(det) > kTiny) || !std::isfinite(det)) {
        return false;
    }
    x0 = (b0 * a11 - a01 * b1) / det;
    x1 = (a00 * b1 - b0 * a10) / det;
    return std::isfinite(x0) && std::isfinite(x1);
}

[[nodiscard]] bool solve3x3(std::array<std::array<Real, 3>, 3> a,
                            std::array<Real, 3> b,
                            std::array<Real, 3>& x) noexcept
{
    for (std::size_t pivot = 0; pivot < 3u; ++pivot) {
        std::size_t best = pivot;
        Real best_abs = std::abs(a[pivot][pivot]);
        for (std::size_t row = pivot + 1u; row < 3u; ++row) {
            const Real value = std::abs(a[row][pivot]);
            if (value > best_abs) {
                best = row;
                best_abs = value;
            }
        }
        if (!(best_abs > kTiny) || !std::isfinite(best_abs)) {
            return false;
        }
        if (best != pivot) {
            std::swap(a[best], a[pivot]);
            std::swap(b[best], b[pivot]);
        }
        const Real inv = Real{1.0} / a[pivot][pivot];
        for (std::size_t col = pivot; col < 3u; ++col) {
            a[pivot][col] *= inv;
        }
        b[pivot] *= inv;
        for (std::size_t row = 0; row < 3u; ++row) {
            if (row == pivot) {
                continue;
            }
            const Real factor = a[row][pivot];
            for (std::size_t col = pivot; col < 3u; ++col) {
                a[row][col] -= factor * a[pivot][col];
            }
            b[row] -= factor * b[pivot];
        }
    }
    x = b;
    return std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]);
}

[[nodiscard]] std::array<Real, 3> centroid(
    const std::vector<std::array<Real, 3>>& points,
    std::size_t count)
{
    std::array<Real, 3> c{{0.0, 0.0, 0.0}};
    if (count == 0u) {
        return c;
    }
    const std::size_t n = std::min(count, points.size());
    for (std::size_t i = 0; i < n; ++i) {
        c = add(c, points[i]);
    }
    return scale(c, Real{1.0} / static_cast<Real>(n));
}

[[nodiscard]] std::array<Real, 3> referenceCentroid(
    const std::vector<SubentityVertex>& vertices)
{
    std::array<Real, 3> c{{0.0, 0.0, 0.0}};
    if (vertices.empty()) {
        return c;
    }
    for (const auto& vertex : vertices) {
        c = add(c, vertex.reference);
    }
    return scale(c, Real{1.0} / static_cast<Real>(vertices.size()));
}

[[nodiscard]] std::array<Real, 3> boundaryNormal(
    int dimension,
    const std::vector<std::array<Real, 3>>& parent_points,
    std::size_t parent_corner_count,
    const std::vector<SubentityVertex>& subentity_vertices)
{
    if (subentity_vertices.size() < 2u) {
        return {{0.0, 1.0, 0.0}};
    }

    const auto cell_center = centroid(parent_points, parent_corner_count);
    const auto face_center = referenceCentroid(subentity_vertices);
    std::array<Real, 3> normal{{0.0, 1.0, 0.0}};
    if (dimension == 2) {
        const auto edge =
            sub(subentity_vertices[1].reference, subentity_vertices[0].reference);
        normal = unitOrDefault({{edge[1], -edge[0], 0.0}},
                               {{0.0, 1.0, 0.0}});
    } else if (subentity_vertices.size() >= 3u) {
        const auto e0 =
            sub(subentity_vertices[1].reference, subentity_vertices[0].reference);
        const auto e1 =
            sub(subentity_vertices[2].reference, subentity_vertices[0].reference);
        normal = unitOrDefault(cross(e0, e1), {{0.0, 0.0, 1.0}});
    }

    if (dot(normal, sub(face_center, cell_center)) < Real{0.0}) {
        normal = scale(normal, Real{-1.0});
    }
    return normal;
}

[[nodiscard]] std::pair<std::array<Real, 3>, Real> estimateImplicitNormal(
    const std::vector<std::array<Real, 3>>& points,
    const std::vector<Real>& signed_values,
    std::size_t count)
{
    if (count == 0u || points.size() < count || signed_values.size() < count) {
        return {{{1.0, 0.0, 0.0}}, Real{0.0}};
    }
    const auto center = centroid(points, count);
    Real value_center = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        value_center += signed_values[i];
    }
    value_center /= static_cast<Real>(count);

    std::array<std::array<Real, 3>, 3> matrix{{
        {{0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}}}};
    std::array<Real, 3> rhs{{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < count; ++i) {
        const auto dx = sub(points[i], center);
        const Real dv = signed_values[i] - value_center;
        for (std::size_t r = 0; r < 3u; ++r) {
            rhs[r] += dx[r] * dv;
            for (std::size_t c = 0; c < 3u; ++c) {
                matrix[r][c] += dx[r] * dx[c];
            }
        }
    }

    std::array<Real, 3> gradient{{0.0, 0.0, 0.0}};
    if (!solve3x3(matrix, rhs, gradient)) {
        std::array<int, 3> axes{{0, 1, 2}};
        std::sort(axes.begin(), axes.end(), [&](int a, int b) {
            return matrix[static_cast<std::size_t>(a)]
                         [static_cast<std::size_t>(a)] >
                   matrix[static_cast<std::size_t>(b)]
                         [static_cast<std::size_t>(b)];
        });
        Real g0 = 0.0;
        Real g1 = 0.0;
        const auto a0 = static_cast<std::size_t>(axes[0]);
        const auto a1 = static_cast<std::size_t>(axes[1]);
        if (solve2x2(matrix[a0][a0],
                     matrix[a0][a1],
                     matrix[a1][a0],
                     matrix[a1][a1],
                     rhs[a0],
                     rhs[a1],
                     g0,
                     g1)) {
            gradient[a0] = g0;
            gradient[a1] = g1;
        }
    }
    const Real gradient_norm = norm(gradient);
    return {unitOrDefault(gradient, {{1.0, 0.0, 0.0}}), gradient_norm};
}

[[nodiscard]] ImplicitBoundaryIntersectionTolerance sanitize(
    ImplicitBoundaryIntersectionTolerance tolerance) noexcept
{
    if (!(tolerance.zero > Real{0.0}) || !std::isfinite(tolerance.zero)) {
        tolerance.zero = Real{1.0e-12};
    }
    if (!(tolerance.duplicate > Real{0.0}) ||
        !std::isfinite(tolerance.duplicate)) {
        tolerance.duplicate = tolerance.zero;
    }
    if (!(tolerance.measure > Real{0.0}) ||
        !std::isfinite(tolerance.measure)) {
        tolerance.measure = tolerance.zero * tolerance.zero;
    }
    return tolerance;
}

[[nodiscard]] bool isZero(Real value, Real tolerance) noexcept
{
    return std::abs(value) <= tolerance;
}

[[nodiscard]] bool oppositeSigns(Real a, Real b) noexcept
{
    return (a < Real{0.0} && b > Real{0.0}) ||
           (a > Real{0.0} && b < Real{0.0});
}

[[nodiscard]] CutCandidate interpolateCandidate(const SubentityVertex& a,
                                                const SubentityVertex& b,
                                                std::size_t edge_index)
{
    const Real denominator = a.signed_value - b.signed_value;
    Real t = Real{0.5};
    if (std::abs(denominator) > kTiny) {
        t = a.signed_value / denominator;
    }
    t = std::clamp(t, Real{0.0}, Real{1.0});
    return CutCandidate{
        .reference = interpolate(a.reference, b.reference, t),
        .physical = interpolate(a.physical, b.physical, t),
        .edge_index = edge_index,
        .at_vertex = false};
}

[[nodiscard]] ImplicitBoundaryIntersectionQuadraturePoint makePoint(
    const CutCandidate& candidate,
    const std::array<Real, 3>& implicit_normal,
    const std::array<Real, 3>& boundary_normal_value,
    const std::array<Real, 3>& tangent_value,
    Real weight,
    Real gradient_norm)
{
    return ImplicitBoundaryIntersectionQuadraturePoint{
        .parent_reference_coordinate = candidate.reference,
        .physical_coordinate = candidate.physical,
        .implicit_normal = implicit_normal,
        .boundary_normal = boundary_normal_value,
        .tangent = tangent_value,
        .weight = weight,
        .scalar_residual = Real{0.0},
        .gradient_norm = gradient_norm};
}

void appendSegmentQuadrature(
    ImplicitBoundaryIntersectionFragment& fragment,
    const ImplicitBoundaryIntersectionRequest& request,
    const CutCandidate& a,
    const CutCandidate& b,
    int requested_order,
    Real gradient_norm)
{
    const auto add_point = [&](Real t, Real weight_fraction) {
        CutCandidate candidate;
        candidate.reference = interpolate(a.reference, b.reference, t);
        candidate.physical =
            request.physical_mapping
                ? request.physical_mapping(candidate.reference)
                : interpolate(a.physical, b.physical, t);
        fragment.quadrature_points.push_back(
            makePoint(candidate,
                      fragment.implicit_normal,
                      fragment.boundary_normal,
                      fragment.tangent,
                      fragment.measure * weight_fraction,
                      gradient_norm));
    };

    if (requested_order <= 1) {
        add_point(Real{0.5}, Real{1.0});
    } else if (requested_order <= 3) {
        constexpr Real offset = Real{0.28867513459481288225};
        add_point(Real{0.5} - offset, Real{0.5});
        add_point(Real{0.5} + offset, Real{0.5});
    } else {
        constexpr Real offset = Real{0.38729833462074168852};
        add_point(Real{0.5} - offset, Real{5.0} / Real{18.0});
        add_point(Real{0.5}, Real{4.0} / Real{9.0});
        add_point(Real{0.5} + offset, Real{5.0} / Real{18.0});
    }
}

[[nodiscard]] ImplicitBoundaryIntersectionFragment makeSegmentFragment(
    const ImplicitBoundaryIntersectionRequest& request,
    const CutCandidate& a,
    const CutCandidate& b,
    ImplicitBoundaryIntersectionStatus status,
    const std::array<Real, 3>& implicit_normal,
    const std::array<Real, 3>& boundary_normal_value,
    Real gradient_norm,
    Real measure_tolerance)
{
    ImplicitBoundaryIntersectionFragment fragment;
    fragment.kind = ImplicitBoundaryIntersectionKind::Segment;
    fragment.status = status;
    fragment.parent_cell = request.parent_cell;
    fragment.local_subentity = request.local_subentity;
    // The returned rule follows the parent-reference contract. Physical
    // length is deliberately not folded into this value; the FE assembler
    // applies ||J t_ref|| exactly once.
    fragment.measure = norm(sub(b.reference, a.reference));
    fragment.implicit_normal = implicit_normal;
    fragment.boundary_normal = boundary_normal_value;
    fragment.tangent = unitOrDefault(sub(b.reference, a.reference),
                                     {{1.0, 0.0, 0.0}});
    fragment.diagnostic = implicitBoundaryIntersectionStatusName(status);
    if (!(fragment.measure > measure_tolerance)) {
        fragment.status = ImplicitBoundaryIntersectionStatus::VanishingMeasure;
        fragment.measure = Real{0.0};
        fragment.diagnostic =
            implicitBoundaryIntersectionStatusName(fragment.status);
        return fragment;
    }
    appendSegmentQuadrature(
        fragment, request, a, b, request.quadrature_order, gradient_norm);
    return fragment;
}

[[nodiscard]] std::vector<SubentityVertex> subentityVertices(
    const ImplicitBoundaryIntersectionRequest& request,
    const std::vector<LocalIndex>& subentity_corners)
{
    std::vector<SubentityVertex> vertices;
    vertices.reserve(subentity_corners.size());
    for (const auto local_node : subentity_corners) {
        if (local_node < 0) {
            throw std::invalid_argument(
                "implicit boundary intersection received a negative subentity node");
        }
        const auto node = static_cast<std::size_t>(local_node);
        if (node >= request.parent_node_coordinates.size()) {
            throw std::invalid_argument(
                "implicit boundary intersection parent coordinates do not cover the selected subentity");
        }
        if (node >= request.scalar_values.size() && !request.scalar_evaluator) {
            throw std::invalid_argument(
                "implicit boundary intersection scalar values do not cover the selected subentity");
        }
        SubentityVertex vertex;
        vertex.local_node = local_node;
        vertex.reference = referenceCoordinate(request.parent_element, node);
        vertex.physical = request.physical_mapping
                              ? request.physical_mapping(vertex.reference)
                              : request.parent_node_coordinates[node];
        const Real scalar = request.scalar_evaluator
                                ? request.scalar_evaluator(vertex.reference)
                                : request.scalar_values[node];
        vertex.signed_value = scalar - request.isovalue;
        vertices.push_back(vertex);
    }
    return vertices;
}

[[nodiscard]] ImplicitBoundaryIntersectionResult invalidResult(
    const ImplicitBoundaryIntersectionRequest& request,
    ImplicitBoundaryIntersectionStatus status,
    std::string diagnostic)
{
    ImplicitBoundaryIntersectionResult result;
    result.parent_element = request.parent_element;
    result.parent_cell = request.parent_cell;
    result.local_subentity = request.local_subentity;
    result.status = status;
    ImplicitBoundaryIntersectionFragment fragment;
    fragment.status = status;
    fragment.parent_cell = request.parent_cell;
    fragment.local_subentity = request.local_subentity;
    fragment.diagnostic = std::move(diagnostic);
    result.fragments.push_back(std::move(fragment));
    return result;
}

} // namespace

bool ImplicitBoundaryIntersectionFragment::active() const noexcept
{
    return status != ImplicitBoundaryIntersectionStatus::Empty &&
           status != ImplicitBoundaryIntersectionStatus::VertexTouch &&
           status != ImplicitBoundaryIntersectionStatus::EdgeAlignedZero &&
           status != ImplicitBoundaryIntersectionStatus::FullyDegenerateSubentity &&
           status != ImplicitBoundaryIntersectionStatus::VanishingMeasure &&
           status != ImplicitBoundaryIntersectionStatus::Ambiguous &&
           status != ImplicitBoundaryIntersectionStatus::UnsupportedElement &&
           status != ImplicitBoundaryIntersectionStatus::InvalidInput &&
           measure > Real{0.0} && !quadrature_points.empty();
}

bool ImplicitBoundaryIntersectionResult::hasActiveFragments() const noexcept
{
    for (const auto& fragment : fragments) {
        if (fragment.active()) {
            return true;
        }
    }
    return false;
}

std::size_t ImplicitBoundaryIntersectionResult::quadraturePointCount()
    const noexcept
{
    std::size_t count = 0u;
    for (const auto& fragment : fragments) {
        if (fragment.active()) {
            count += fragment.quadrature_points.size();
        }
    }
    return count;
}

Real ImplicitBoundaryIntersectionResult::measure() const noexcept
{
    Real value = 0.0;
    for (const auto& fragment : fragments) {
        if (fragment.active()) {
            value += fragment.measure;
        }
    }
    return value;
}

const char* implicitBoundaryIntersectionStatusName(
    ImplicitBoundaryIntersectionStatus status) noexcept
{
    switch (status) {
    case ImplicitBoundaryIntersectionStatus::Empty:
        return "empty";
    case ImplicitBoundaryIntersectionStatus::Active:
        return "active";
    case ImplicitBoundaryIntersectionStatus::VertexTouch:
        return "vertex_touch";
    case ImplicitBoundaryIntersectionStatus::EdgeAlignedZero:
        return "edge_aligned_zero";
    case ImplicitBoundaryIntersectionStatus::FullyDegenerateSubentity:
        return "fully_degenerate_subentity";
    case ImplicitBoundaryIntersectionStatus::VanishingMeasure:
        return "vanishing_measure";
    case ImplicitBoundaryIntersectionStatus::Ambiguous:
        return "ambiguous";
    case ImplicitBoundaryIntersectionStatus::UnsupportedElement:
        return "unsupported_element";
    case ImplicitBoundaryIntersectionStatus::InvalidInput:
        return "invalid_input";
    }
    return "unknown";
}

bool supportsImplicitBoundaryIntersectionQuadrature(
    ElementType parent_element) noexcept
{
    // High-order parents cannot be made accurate by merely increasing the
    // quadrature order on a corner-linear reconstruction. Reject them until a
    // curved/isoparametric intersection path is supplied.
    switch (parent_element) {
    case ElementType::Triangle3:
    case ElementType::Quad4:
    case ElementType::Tetra4:
    case ElementType::Hex8:
    case ElementType::Wedge6:
    case ElementType::Pyramid5:
        return true;
    default:
        return false;
    }
}

std::vector<LocalIndex> implicitBoundarySubentityCornerIndices(
    ElementType parent_element,
    LocalIndex local_subentity)
{
    if (local_subentity < 0) {
        return {};
    }
    if (!supportsImplicitBoundaryIntersectionQuadrature(parent_element)) {
        return {};
    }
    const auto ref = elements::ReferenceElement::create(parent_element);
    const auto subentity = static_cast<std::size_t>(local_subentity);
    if (subentity >= ref.num_faces()) {
        return {};
    }
    return ref.face_nodes(subentity);
}

ImplicitBoundaryIntersectionResult buildImplicitBoundaryIntersectionQuadrature(
    const ImplicitBoundaryIntersectionRequest& request)
{
    ImplicitBoundaryIntersectionResult result;
    result.parent_element = request.parent_element;
    result.parent_cell = request.parent_cell;
    result.local_subentity = request.local_subentity;
    result.status = ImplicitBoundaryIntersectionStatus::Empty;

    if (!supportsImplicitBoundaryIntersectionQuadrature(request.parent_element)) {
        return invalidResult(
            request,
            ImplicitBoundaryIntersectionStatus::UnsupportedElement,
            "parent element is not supported by implicit boundary intersection quadrature");
    }
    if (request.local_subentity < 0 || request.quadrature_order < 0) {
        return invalidResult(
            request,
            ImplicitBoundaryIntersectionStatus::InvalidInput,
            "implicit boundary intersection request has an invalid subentity or quadrature order");
    }
    const int dimension = request.parent_dimension > 0
                              ? request.parent_dimension
                              : element_dimension(request.parent_element);
    if (dimension != 2 && dimension != 3) {
        return invalidResult(
            request,
            ImplicitBoundaryIntersectionStatus::UnsupportedElement,
            "implicit boundary intersection requires a 2D or 3D parent");
    }

    const auto parent_corner_count = cornerCount(request.parent_element);
    if (parent_corner_count == 0u ||
        request.parent_node_coordinates.size() < parent_corner_count ||
        (request.scalar_values.size() < parent_corner_count &&
         !request.scalar_evaluator)) {
        return invalidResult(
            request,
            ImplicitBoundaryIntersectionStatus::InvalidInput,
            "implicit boundary intersection request is missing parent corner data");
    }
    const auto tolerance = sanitize(request.tolerance);
    const auto subentity_corners =
        implicitBoundarySubentityCornerIndices(request.parent_element,
                                              request.local_subentity);
    if (subentity_corners.empty()) {
        return invalidResult(
            request,
            ImplicitBoundaryIntersectionStatus::InvalidInput,
            "implicit boundary intersection request references an unknown boundary subentity");
    }
    const auto vertices = subentityVertices(request, subentity_corners);
    if (vertices.size() < 2u) {
        return invalidResult(
            request,
            ImplicitBoundaryIntersectionStatus::InvalidInput,
            "implicit boundary intersection subentity has too few vertices");
    }

    std::vector<Real> parent_signed_values(parent_corner_count, 0.0);
    std::vector<std::array<Real, 3>> parent_reference_points;
    parent_reference_points.reserve(parent_corner_count);
    for (std::size_t i = 0; i < parent_corner_count; ++i) {
        const auto ref = referenceCoordinate(request.parent_element, i);
        parent_reference_points.push_back(ref);
        const Real scalar = request.scalar_evaluator
                                ? request.scalar_evaluator(ref)
                                : request.scalar_values[i];
        parent_signed_values[i] = scalar - request.isovalue;
    }
    const auto normal_estimate =
        estimateImplicitNormal(parent_reference_points,
                               parent_signed_values,
                               parent_corner_count);
    const auto implicit_normal = normal_estimate.first;
    const Real gradient_norm = normal_estimate.second;
    const auto outward =
        boundaryNormal(dimension,
                       parent_reference_points,
                       parent_corner_count,
                       vertices);
    const auto default_tangent =
        unitOrDefault(sub(vertices[1].reference, vertices[0].reference),
                      {{1.0, 0.0, 0.0}});

    bool all_zero = true;
    bool all_positive = true;
    bool all_negative = true;
    for (const auto& vertex : vertices) {
        const bool zero = isZero(vertex.signed_value, tolerance.zero);
        all_zero = all_zero && zero;
        all_positive = all_positive && vertex.signed_value > tolerance.zero;
        all_negative = all_negative && vertex.signed_value < -tolerance.zero;
    }
    if (all_positive || all_negative) {
        result.status = ImplicitBoundaryIntersectionStatus::Empty;
        return result;
    }
    if (all_zero) {
        result.status = dimension == 2
                            ? ImplicitBoundaryIntersectionStatus::EdgeAlignedZero
                            : ImplicitBoundaryIntersectionStatus::
                                  FullyDegenerateSubentity;
        ImplicitBoundaryIntersectionFragment fragment;
        fragment.status = result.status;
        fragment.parent_cell = request.parent_cell;
        fragment.local_subentity = request.local_subentity;
        fragment.implicit_normal = implicit_normal;
        fragment.boundary_normal = outward;
        fragment.tangent = default_tangent;
        fragment.diagnostic = implicitBoundaryIntersectionStatusName(result.status);
        result.fragments.push_back(std::move(fragment));
        return result;
    }

    std::vector<CutCandidate> candidates;
    std::vector<std::pair<CutCandidate, CutCandidate>> aligned_segments;
    const std::size_t edge_count = dimension == 2 ? 1u : vertices.size();
    for (std::size_t edge = 0; edge < edge_count; ++edge) {
        const auto& a = vertices[edge];
        const auto& b =
            vertices[dimension == 2 ? 1u : (edge + 1u) % vertices.size()];
        const bool a_zero = isZero(a.signed_value, tolerance.zero);
        const bool b_zero = isZero(b.signed_value, tolerance.zero);
        if (a_zero && b_zero) {
            aligned_segments.push_back(
                {CutCandidate{.reference = a.reference,
                              .physical = a.physical,
                              .edge_index = edge,
                              .at_vertex = true},
                 CutCandidate{.reference = b.reference,
                              .physical = b.physical,
                              .edge_index = edge,
                              .at_vertex = true}});
        } else if (a_zero) {
            addUniqueCandidate(candidates,
                               CutCandidate{.reference = a.reference,
                                            .physical = a.physical,
                                            .edge_index = edge,
                                            .at_vertex = true},
                               tolerance.duplicate);
        } else if (b_zero) {
            addUniqueCandidate(candidates,
                               CutCandidate{.reference = b.reference,
                                            .physical = b.physical,
                                            .edge_index = edge,
                                            .at_vertex = true},
                               tolerance.duplicate);
        } else if (oppositeSigns(a.signed_value, b.signed_value)) {
            addUniqueCandidate(candidates,
                               interpolateCandidate(a, b, edge),
                               tolerance.duplicate);
        }
    }

    if (dimension == 2) {
        if (!aligned_segments.empty()) {
            result.status = ImplicitBoundaryIntersectionStatus::EdgeAlignedZero;
            ImplicitBoundaryIntersectionFragment fragment;
            fragment.kind = ImplicitBoundaryIntersectionKind::Point;
            fragment.status = result.status;
            fragment.parent_cell = request.parent_cell;
            fragment.local_subentity = request.local_subentity;
            fragment.implicit_normal = implicit_normal;
            fragment.boundary_normal = outward;
            fragment.tangent = default_tangent;
            fragment.diagnostic =
                "implicit field is zero along the selected boundary edge";
            result.fragments.push_back(std::move(fragment));
            return result;
        }
        if (candidates.empty()) {
            result.status = ImplicitBoundaryIntersectionStatus::Empty;
            return result;
        }
        if (candidates.size() != 1u) {
            result.status = ImplicitBoundaryIntersectionStatus::Ambiguous;
            ImplicitBoundaryIntersectionFragment fragment;
            fragment.kind = ImplicitBoundaryIntersectionKind::Point;
            fragment.status = result.status;
            fragment.parent_cell = request.parent_cell;
            fragment.local_subentity = request.local_subentity;
            fragment.implicit_normal = implicit_normal;
            fragment.boundary_normal = outward;
            fragment.tangent = default_tangent;
            fragment.diagnostic =
                "multiple implicit roots were found on one boundary edge";
            result.fragments.push_back(std::move(fragment));
            return result;
        }
        if (candidates.front().at_vertex) {
            result.status = ImplicitBoundaryIntersectionStatus::VertexTouch;
            ImplicitBoundaryIntersectionFragment fragment;
            fragment.kind = ImplicitBoundaryIntersectionKind::Point;
            fragment.status = result.status;
            fragment.parent_cell = request.parent_cell;
            fragment.local_subentity = request.local_subentity;
            fragment.implicit_normal = implicit_normal;
            fragment.boundary_normal = outward;
            fragment.tangent = default_tangent;
            fragment.diagnostic =
                "implicit contact point coincides with a boundary vertex; ownership is unresolved";
            result.fragments.push_back(std::move(fragment));
            return result;
        }
        result.status = ImplicitBoundaryIntersectionStatus::Active;
        auto candidate = candidates.front();
        ImplicitBoundaryIntersectionFragment fragment;
        fragment.kind = ImplicitBoundaryIntersectionKind::Point;
        fragment.status = result.status;
        fragment.parent_cell = request.parent_cell;
        fragment.local_subentity = request.local_subentity;
        fragment.measure = Real{1.0};
        fragment.implicit_normal = implicit_normal;
        fragment.boundary_normal = outward;
        fragment.tangent = default_tangent;
        fragment.diagnostic = implicitBoundaryIntersectionStatusName(result.status);
        fragment.quadrature_points.push_back(
            makePoint(candidate,
                      implicit_normal,
                      outward,
                      default_tangent,
                      fragment.measure,
                      gradient_norm));
        result.fragments.push_back(std::move(fragment));
        return result;
    }

    if (!aligned_segments.empty()) {
        // An interface coincident with a boundary-face edge is shared by
        // multiple subentities/cells. Until global ownership is available,
        // accepting it would silently double-count the contact set.
        result.status = ImplicitBoundaryIntersectionStatus::EdgeAlignedZero;
        ImplicitBoundaryIntersectionFragment fragment;
        fragment.kind = ImplicitBoundaryIntersectionKind::Segment;
        fragment.status = result.status;
        fragment.parent_cell = request.parent_cell;
        fragment.local_subentity = request.local_subentity;
        fragment.implicit_normal = implicit_normal;
        fragment.boundary_normal = outward;
        fragment.tangent = default_tangent;
        fragment.diagnostic =
            "implicit intersection is aligned with a boundary-face edge; global ownership is unresolved";
        result.fragments.push_back(std::move(fragment));
        return result;
    }

    std::sort(candidates.begin(),
              candidates.end(),
              [](const CutCandidate& a, const CutCandidate& b) noexcept {
                  if (a.edge_index != b.edge_index) {
                      return a.edge_index < b.edge_index;
                  }
                  if (a.reference[0] != b.reference[0]) {
                      return a.reference[0] < b.reference[0];
                  }
                  if (a.reference[1] != b.reference[1]) {
                      return a.reference[1] < b.reference[1];
                  }
                  return a.reference[2] < b.reference[2];
              });

    if (candidates.size() == 1u && aligned_segments.empty()) {
        result.status = ImplicitBoundaryIntersectionStatus::VertexTouch;
        ImplicitBoundaryIntersectionFragment fragment;
        fragment.kind = ImplicitBoundaryIntersectionKind::Point;
        fragment.status = ImplicitBoundaryIntersectionStatus::VertexTouch;
        fragment.parent_cell = request.parent_cell;
        fragment.local_subentity = request.local_subentity;
        fragment.implicit_normal = implicit_normal;
        fragment.boundary_normal = outward;
        fragment.tangent = default_tangent;
        fragment.diagnostic = "implicit field touches the selected boundary face at one point";
        result.fragments.push_back(std::move(fragment));
        return result;
    }

    const auto append_candidate_segment =
        [&](std::size_t a_index,
            std::size_t b_index,
            ImplicitBoundaryIntersectionStatus status) {
            auto fragment =
                makeSegmentFragment(request,
                                    candidates[a_index],
                                    candidates[b_index],
                                    status,
                                    implicit_normal,
                                    outward,
                                    gradient_norm,
                                    tolerance.measure);
            result.fragments.push_back(std::move(fragment));
        };

    if (candidates.size() == 2u) {
        append_candidate_segment(0u, 1u, ImplicitBoundaryIntersectionStatus::Active);
    } else if (candidates.size() > 2u) {
        // A bilinear quadrilateral face can have four roots (a saddle). Edge
        // order does not determine the correct branch connectivity. Preserve
        // the ambiguity but do not manufacture and assemble arbitrary pairs.
        result.status = ImplicitBoundaryIntersectionStatus::Ambiguous;
        ImplicitBoundaryIntersectionFragment fragment;
        fragment.kind = ImplicitBoundaryIntersectionKind::Segment;
        fragment.status = result.status;
        fragment.parent_cell = request.parent_cell;
        fragment.local_subentity = request.local_subentity;
        fragment.implicit_normal = implicit_normal;
        fragment.boundary_normal = outward;
        fragment.tangent = default_tangent;
        fragment.diagnostic =
            "boundary face has more than two implicit roots; topology requires an asymptotic decider";
        result.fragments.push_back(std::move(fragment));
        return result;
    }

    result.status = result.hasActiveFragments()
                        ? ImplicitBoundaryIntersectionStatus::Active
                        : ImplicitBoundaryIntersectionStatus::Empty;
    if (result.fragments.empty()) {
        result.status = ImplicitBoundaryIntersectionStatus::Empty;
    }
    return result;
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
