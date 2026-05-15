/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace svmp {
namespace FE {
namespace interfaces {
namespace {

struct CutPointCandidate {
    std::array<Real, 3> point{{0.0, 0.0, 0.0}};
    std::array<Real, 3> parent_coordinate{{0.0, 0.0, 0.0}};
    Real level_set_value{0.0};
};

struct SignedPoint {
    std::array<Real, 3> point{{0.0, 0.0, 0.0}};
    Real value{0.0};
};

struct RegionMoments {
    Real measure{0.0};
    std::array<Real, 3> centroid{{0.0, 0.0, 0.0}};
};

[[nodiscard]] Real dot2(const std::array<Real, 3>& a,
                        const std::array<Real, 3>& b) noexcept {
    return a[0] * b[0] + a[1] * b[1];
}

[[nodiscard]] Real dot3(const std::array<Real, 3>& a,
                        const std::array<Real, 3>& b) noexcept {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] std::array<Real, 3> sub(const std::array<Real, 3>& a,
                                      const std::array<Real, 3>& b) noexcept {
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

[[nodiscard]] std::array<Real, 3> add(const std::array<Real, 3>& a,
                                      const std::array<Real, 3>& b) noexcept {
    return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
}

[[nodiscard]] std::array<Real, 3> scale(const std::array<Real, 3>& a,
                                        Real s) noexcept {
    return {{a[0] * s, a[1] * s, a[2] * s}};
}

[[nodiscard]] Real norm2(const std::array<Real, 3>& a) noexcept {
    return std::sqrt(dot2(a, a));
}

[[nodiscard]] Real norm3(const std::array<Real, 3>& a) noexcept {
    return std::sqrt(dot3(a, a));
}

[[nodiscard]] std::array<Real, 3> cross(const std::array<Real, 3>& a,
                                        const std::array<Real, 3>& b) noexcept {
    return {{a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]}};
}

[[nodiscard]] std::array<Real, 3> unitOrDefault(const std::array<Real, 3>& a,
                                                const std::array<Real, 3>& fallback) noexcept {
    const Real len = norm3(a);
    if (len <= Real{1.0e-30}) {
        return fallback;
    }
    return scale(a, Real{1.0} / len);
}

[[nodiscard]] Real distance2(const std::array<Real, 3>& a,
                             const std::array<Real, 3>& b) noexcept {
    return norm2(sub(a, b));
}

[[nodiscard]] bool nearlySamePoint(const std::array<Real, 3>& a,
                                   const std::array<Real, 3>& b,
                                   Real tolerance) noexcept {
    return norm3(sub(a, b)) <= tolerance;
}

[[nodiscard]] std::size_t cornerCount(ElementType element_type) {
    switch (element_type) {
    case ElementType::Triangle3:
    case ElementType::Triangle6:
        return 3u;
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
        return 4u;
    default:
        throw std::invalid_argument("linear level-set 2D cutter requires a triangle or quadrilateral");
    }
}

[[nodiscard]] std::size_t cornerCount3D(ElementType element_type) {
    switch (element_type) {
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return 4u;
    default:
        throw std::invalid_argument("linear level-set 3D cutter requires a tetrahedron");
    }
}

[[nodiscard]] std::array<Real, 3> interpolate(const std::array<Real, 3>& a,
                                              const std::array<Real, 3>& b,
                                              Real t) noexcept {
    return add(scale(a, Real{1.0} - t), scale(b, t));
}

[[nodiscard]] std::array<Real, 3> barycentricPoint(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b,
    const std::array<Real, 3>& c,
    Real wa,
    Real wb,
    Real wc) noexcept {
    return add(add(scale(a, wa), scale(b, wb)), scale(c, wc));
}

[[nodiscard]] std::array<Real, 3> barycentricPoint(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b,
    const std::array<Real, 3>& c,
    const std::array<Real, 3>& d,
    Real wa,
    Real wb,
    Real wc,
    Real wd) noexcept {
    return add(add(scale(a, wa), scale(b, wb)),
               add(scale(c, wc), scale(d, wd)));
}

[[nodiscard]] CutInterfaceDegeneracy classifyZeroContact2D(
    const std::vector<Real>& signed_values,
    std::size_t count,
    Real tolerance) noexcept
{
    bool has_zero = false;
    for (std::size_t i = 0; i < count; ++i) {
        has_zero = has_zero || std::abs(signed_values[i]) <= tolerance;
    }
    if (!has_zero) {
        return CutInterfaceDegeneracy::NoCut;
    }
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t j = (i + 1u) % count;
        if (std::abs(signed_values[i]) <= tolerance &&
            std::abs(signed_values[j]) <= tolerance) {
            return CutInterfaceDegeneracy::EdgeTouch;
        }
    }
    return CutInterfaceDegeneracy::VertexTouch;
}

[[nodiscard]] CutInterfaceDegeneracy classifyZeroContactTetrahedron(
    const std::vector<Real>& signed_values,
    Real tolerance) noexcept
{
    bool has_zero = false;
    for (const Real value : signed_values) {
        has_zero = has_zero || std::abs(value) <= tolerance;
    }
    if (!has_zero) {
        return CutInterfaceDegeneracy::NoCut;
    }
    constexpr std::array<std::array<std::size_t, 2>, 6> edges{{
        {{0u, 1u}},
        {{0u, 2u}},
        {{0u, 3u}},
        {{1u, 2u}},
        {{1u, 3u}},
        {{2u, 3u}}}};
    for (const auto& edge : edges) {
        if (std::abs(signed_values[edge[0]]) <= tolerance &&
            std::abs(signed_values[edge[1]]) <= tolerance) {
            return CutInterfaceDegeneracy::EdgeTouch;
        }
    }
    return CutInterfaceDegeneracy::VertexTouch;
}

void addUniqueCandidate(std::vector<CutPointCandidate>& points,
                        CutPointCandidate candidate,
                        Real tolerance) {
    const auto duplicate = std::find_if(
        points.begin(),
        points.end(),
        [&](const CutPointCandidate& existing) {
            return nearlySamePoint(existing.point, candidate.point, tolerance);
        });
    if (duplicate == points.end()) {
        points.push_back(std::move(candidate));
    }
}

[[nodiscard]] bool solve3x3(std::array<std::array<Real, 3>, 3> matrix,
                            std::array<Real, 3> rhs,
                            std::array<Real, 3>& solution) noexcept {
    for (std::size_t pivot = 0; pivot < 3u; ++pivot) {
        std::size_t best = pivot;
        Real best_abs = std::abs(matrix[pivot][pivot]);
        for (std::size_t row = pivot + 1u; row < 3u; ++row) {
            const Real value = std::abs(matrix[row][pivot]);
            if (value > best_abs) {
                best = row;
                best_abs = value;
            }
        }
        if (best_abs <= Real{1.0e-30}) {
            return false;
        }
        if (best != pivot) {
            std::swap(matrix[best], matrix[pivot]);
            std::swap(rhs[best], rhs[pivot]);
        }
        const Real inv_pivot = Real{1.0} / matrix[pivot][pivot];
        for (std::size_t col = pivot; col < 3u; ++col) {
            matrix[pivot][col] *= inv_pivot;
        }
        rhs[pivot] *= inv_pivot;
        for (std::size_t row = 0; row < 3u; ++row) {
            if (row == pivot) {
                continue;
            }
            const Real factor = matrix[row][pivot];
            for (std::size_t col = pivot; col < 3u; ++col) {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    solution = rhs;
    return true;
}

[[nodiscard]] bool solve4x4(std::array<std::array<Real, 4>, 4> matrix,
                            std::array<Real, 4> rhs,
                            std::array<Real, 4>& solution) noexcept {
    for (std::size_t pivot = 0; pivot < 4u; ++pivot) {
        std::size_t best = pivot;
        Real best_abs = std::abs(matrix[pivot][pivot]);
        for (std::size_t row = pivot + 1u; row < 4u; ++row) {
            const Real value = std::abs(matrix[row][pivot]);
            if (value > best_abs) {
                best = row;
                best_abs = value;
            }
        }
        if (best_abs <= Real{1.0e-30}) {
            return false;
        }
        if (best != pivot) {
            std::swap(matrix[best], matrix[pivot]);
            std::swap(rhs[best], rhs[pivot]);
        }
        const Real inv_pivot = Real{1.0} / matrix[pivot][pivot];
        for (std::size_t col = pivot; col < 4u; ++col) {
            matrix[pivot][col] *= inv_pivot;
        }
        rhs[pivot] *= inv_pivot;
        for (std::size_t row = 0; row < 4u; ++row) {
            if (row == pivot) {
                continue;
            }
            const Real factor = matrix[row][pivot];
            for (std::size_t col = pivot; col < 4u; ++col) {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    solution = rhs;
    return true;
}

[[nodiscard]] std::array<Real, 3> estimateGradient2D(
    const std::vector<std::array<Real, 3>>& points,
    const std::vector<Real>& signed_values,
    std::size_t count) noexcept {
    std::array<std::array<Real, 3>, 3> normal_matrix{{
        {{0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}}}};
    std::array<Real, 3> rhs{{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < count; ++i) {
        const std::array<Real, 3> row{{points[i][0], points[i][1], 1.0}};
        for (std::size_t r = 0; r < 3u; ++r) {
            rhs[r] += row[r] * signed_values[i];
            for (std::size_t c = 0; c < 3u; ++c) {
                normal_matrix[r][c] += row[r] * row[c];
            }
        }
    }
    std::array<Real, 3> solution{{0.0, 0.0, 0.0}};
    if (!solve3x3(normal_matrix, rhs, solution)) {
        return {{1.0, 0.0, 0.0}};
    }
    const Real len = std::sqrt(solution[0] * solution[0] + solution[1] * solution[1]);
    if (len <= Real{1.0e-30}) {
        return {{1.0, 0.0, 0.0}};
    }
    return {{solution[0] / len, solution[1] / len, 0.0}};
}

[[nodiscard]] std::array<Real, 3> estimateGradient3D(
    const std::vector<std::array<Real, 3>>& points,
    const std::vector<Real>& signed_values,
    std::size_t count) noexcept {
    std::array<std::array<Real, 4>, 4> matrix{{
        {{0.0, 0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0, 0.0}}}};
    std::array<Real, 4> rhs{{0.0, 0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < count; ++i) {
        const std::array<Real, 4> row{{points[i][0], points[i][1], points[i][2], 1.0}};
        for (std::size_t r = 0; r < 4u; ++r) {
            rhs[r] += row[r] * signed_values[i];
            for (std::size_t c = 0; c < 4u; ++c) {
                matrix[r][c] += row[r] * row[c];
            }
        }
    }
    std::array<Real, 4> solution{{0.0, 0.0, 0.0, 0.0}};
    if (!solve4x4(matrix, rhs, solution)) {
        return {{1.0, 0.0, 0.0}};
    }
    return unitOrDefault({{solution[0], solution[1], solution[2]}},
                         {{1.0, 0.0, 0.0}});
}

[[nodiscard]] std::pair<std::size_t, std::size_t> farthestPair(
    const std::vector<CutPointCandidate>& points) noexcept {
    std::pair<std::size_t, std::size_t> pair{0u, 1u};
    Real max_distance = Real{-1.0};
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (std::size_t j = i + 1u; j < points.size(); ++j) {
            const Real d = distance2(points[i].point, points[j].point);
            if (d > max_distance) {
                max_distance = d;
                pair = {i, j};
            }
        }
    }
    return pair;
}

[[nodiscard]] std::array<Real, 3> centroid(
    const std::vector<CutPointCandidate>& points) noexcept {
    std::array<Real, 3> c{{0.0, 0.0, 0.0}};
    if (points.empty()) {
        return c;
    }
    for (const auto& point : points) {
        c = add(c, point.point);
    }
    return scale(c, Real{1.0} / static_cast<Real>(points.size()));
}

void orderPolygonPoints(std::vector<CutPointCandidate>& points,
                        const std::array<Real, 3>& normal) {
    if (points.size() < 3u) {
        return;
    }
    const auto c = centroid(points);
    std::array<Real, 3> axis0 = unitOrDefault(sub(points.front().point, c),
                                              {{1.0, 0.0, 0.0}});
    if (norm3(cross(axis0, normal)) <= Real{1.0e-30}) {
        axis0 = unitOrDefault(cross(normal, {{0.0, 1.0, 0.0}}),
                              {{1.0, 0.0, 0.0}});
    }
    const auto axis1 = unitOrDefault(cross(normal, axis0),
                                     {{0.0, 1.0, 0.0}});
    std::sort(points.begin(),
              points.end(),
              [&](const CutPointCandidate& a, const CutPointCandidate& b) {
                  const auto da = sub(a.point, c);
                  const auto db = sub(b.point, c);
                  const Real angle_a = std::atan2(dot3(da, axis1), dot3(da, axis0));
                  const Real angle_b = std::atan2(dot3(db, axis1), dot3(db, axis0));
                  return angle_a < angle_b;
              });
}

[[nodiscard]] Real polygonArea(const std::vector<CutPointCandidate>& points,
                               const std::array<Real, 3>& normal) noexcept {
    if (points.size() < 3u) {
        return Real{0.0};
    }
    const auto c = centroid(points);
    Real area = Real{0.0};
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto a = sub(points[i].point, c);
        const auto b = sub(points[(i + 1u) % points.size()].point, c);
        area += Real{0.5} * std::abs(dot3(cross(a, b), normal));
    }
    return area;
}

[[nodiscard]] Real clampFraction(Real value) noexcept {
    return std::max(Real{0.0}, std::min(Real{1.0}, value));
}

[[nodiscard]] SignedPoint interpolateSignedPoint(const SignedPoint& a,
                                                 const SignedPoint& b) noexcept {
    const Real denominator = a.value - b.value;
    Real t = Real{0.0};
    if (std::abs(denominator) > Real{1.0e-30}) {
        t = clampFraction(a.value / denominator);
    }
    return SignedPoint{interpolate(a.point, b.point, t), Real{0.0}};
}

[[nodiscard]] std::vector<SignedPoint> clipPolygonToNegativeLevelSet(
    const std::vector<SignedPoint>& polygon,
    Real tolerance)
{
    std::vector<SignedPoint> clipped;
    if (polygon.empty()) {
        return clipped;
    }

    const auto inside = [tolerance](const SignedPoint& point) noexcept {
        return point.value <= tolerance;
    };
    SignedPoint previous = polygon.back();
    bool previous_inside = inside(previous);
    for (const auto& current : polygon) {
        const bool current_inside = inside(current);
        if (previous_inside && current_inside) {
            clipped.push_back(current);
        } else if (previous_inside && !current_inside) {
            clipped.push_back(interpolateSignedPoint(previous, current));
        } else if (!previous_inside && current_inside) {
            clipped.push_back(interpolateSignedPoint(previous, current));
            clipped.push_back(current);
        }
        previous = current;
        previous_inside = current_inside;
    }
    return clipped;
}

[[nodiscard]] std::vector<SignedPoint> clipPolygonToPositiveLevelSet(
    const std::vector<SignedPoint>& polygon,
    Real tolerance)
{
    std::vector<SignedPoint> clipped;
    if (polygon.empty()) {
        return clipped;
    }

    const auto inside = [tolerance](const SignedPoint& point) noexcept {
        return point.value >= -tolerance;
    };
    SignedPoint previous = polygon.back();
    bool previous_inside = inside(previous);
    for (const auto& current : polygon) {
        const bool current_inside = inside(current);
        if (previous_inside && current_inside) {
            clipped.push_back(current);
        } else if (previous_inside && !current_inside) {
            clipped.push_back(interpolateSignedPoint(previous, current));
        } else if (!previous_inside && current_inside) {
            clipped.push_back(interpolateSignedPoint(previous, current));
            clipped.push_back(current);
        }
        previous = current;
        previous_inside = current_inside;
    }
    return clipped;
}

[[nodiscard]] RegionMoments polygonMoments2D(
    const std::vector<SignedPoint>& polygon) noexcept
{
    RegionMoments moments;
    if (polygon.size() < 3u) {
        return moments;
    }

    const auto& origin = polygon.front().point;
    Real signed_measure = Real{0.0};
    std::array<Real, 3> first_moment{{0.0, 0.0, 0.0}};
    for (std::size_t i = 1u; i + 1u < polygon.size(); ++i) {
        const auto& a = origin;
        const auto& b = polygon[i].point;
        const auto& c = polygon[i + 1u].point;
        const Real triangle_measure =
            Real{0.5} * ((b[0] - a[0]) * (c[1] - a[1]) -
                         (b[1] - a[1]) * (c[0] - a[0]));
        signed_measure += triangle_measure;
        const auto triangle_centroid =
            scale(add(add(a, b), c), Real{1.0} / Real{3.0});
        first_moment = add(first_moment,
                           scale(triangle_centroid, triangle_measure));
    }
    if (std::abs(signed_measure) <= Real{1.0e-30}) {
        return RegionMoments{};
    }
    moments.measure = std::abs(signed_measure);
    moments.centroid = scale(first_moment, Real{1.0} / signed_measure);
    return moments;
}

[[nodiscard]] std::vector<SignedPoint> makeSignedPolygon(
    const std::vector<std::array<Real, 3>>& points,
    const std::vector<Real>& signed_values,
    std::size_t count)
{
    std::vector<SignedPoint> polygon;
    polygon.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        polygon.push_back(SignedPoint{points[i], signed_values[i]});
    }
    return polygon;
}

[[nodiscard]] RegionMoments parentMoments2D(
    const std::vector<std::array<Real, 3>>& points,
    std::size_t count)
{
    std::vector<SignedPoint> polygon;
    polygon.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        polygon.push_back(SignedPoint{points[i], Real{0.0}});
    }
    return polygonMoments2D(polygon);
}

[[nodiscard]] std::vector<SignedPoint> cutSidePolygon2D(
    const std::vector<std::array<Real, 3>>& points,
    const std::vector<Real>& signed_values,
    std::size_t count,
    geometry::CutIntegrationSide side,
    Real tolerance)
{
    const auto polygon = makeSignedPolygon(points, signed_values, count);
    return side == geometry::CutIntegrationSide::Negative
               ? clipPolygonToNegativeLevelSet(polygon, tolerance)
               : clipPolygonToPositiveLevelSet(polygon, tolerance);
}

[[nodiscard]] RegionMoments cutSideMoments2D(
    const std::vector<std::array<Real, 3>>& points,
    const std::vector<Real>& signed_values,
    std::size_t count,
    geometry::CutIntegrationSide side,
    Real tolerance)
{
    return polygonMoments2D(
        cutSidePolygon2D(points, signed_values, count, side, tolerance));
}

[[nodiscard]] std::vector<geometry::CutQuadraturePoint> polygonQuadrature2D(
    const std::vector<SignedPoint>& polygon,
    Real tolerance)
{
    std::vector<geometry::CutQuadraturePoint> points;
    if (polygon.size() < 3u) {
        return points;
    }

    const auto& origin = polygon.front().point;
    points.reserve(polygon.size() - 2u);
    for (std::size_t i = 1u; i + 1u < polygon.size(); ++i) {
        const auto& b = polygon[i].point;
        const auto& c = polygon[i + 1u].point;
        const Real area =
            Real{0.5} * norm3(cross(sub(b, origin), sub(c, origin)));
        if (area <= tolerance) {
            continue;
        }
        constexpr Real high = Real{2.0} / Real{3.0};
        constexpr Real low = Real{1.0} / Real{6.0};
        const Real weight = area / Real{3.0};
        points.push_back(geometry::CutQuadraturePoint{
            .point = barycentricPoint(origin, b, c, high, low, low),
            .weight = weight});
        points.push_back(geometry::CutQuadraturePoint{
            .point = barycentricPoint(origin, b, c, low, high, low),
            .weight = weight});
        points.push_back(geometry::CutQuadraturePoint{
            .point = barycentricPoint(origin, b, c, low, low, high),
            .weight = weight});
    }
    return points;
}

[[nodiscard]] std::vector<geometry::CutQuadraturePoint> cutSideQuadrature2D(
    const std::vector<std::array<Real, 3>>& points,
    const std::vector<Real>& signed_values,
    std::size_t count,
    geometry::CutIntegrationSide side,
    Real tolerance)
{
    return polygonQuadrature2D(
        cutSidePolygon2D(points, signed_values, count, side, tolerance),
        tolerance);
}

void addUniquePoint(std::vector<std::array<Real, 3>>& points,
                    const std::array<Real, 3>& point,
                    Real tolerance)
{
    const auto duplicate = std::find_if(
        points.begin(),
        points.end(),
        [&](const std::array<Real, 3>& existing) {
            return nearlySamePoint(existing, point, tolerance);
        });
    if (duplicate == points.end()) {
        points.push_back(point);
    }
}

[[nodiscard]] Real tetraVolume(const std::array<Real, 3>& a,
                               const std::array<Real, 3>& b,
                               const std::array<Real, 3>& c,
                               const std::array<Real, 3>& d) noexcept {
    return std::abs(dot3(sub(b, a), cross(sub(c, a), sub(d, a)))) / Real{6.0};
}

[[nodiscard]] RegionMoments polyhedronMomentsFromFaces(
    const std::vector<std::vector<std::array<Real, 3>>>& faces,
    Real tolerance)
{
    std::vector<std::array<Real, 3>> unique_points;
    for (const auto& face : faces) {
        for (const auto& point : face) {
            addUniquePoint(unique_points, point, tolerance);
        }
    }
    if (unique_points.empty()) {
        return RegionMoments{};
    }

    std::array<Real, 3> center{{0.0, 0.0, 0.0}};
    for (const auto& point : unique_points) {
        center = add(center, point);
    }
    center = scale(center, Real{1.0} / static_cast<Real>(unique_points.size()));

    RegionMoments moments;
    std::array<Real, 3> first_moment{{0.0, 0.0, 0.0}};
    for (const auto& face : faces) {
        if (face.size() < 3u) {
            continue;
        }
        for (std::size_t i = 1u; i + 1u < face.size(); ++i) {
            const Real volume = std::abs(dot3(sub(face[0], center),
                                              cross(sub(face[i], center),
                                                    sub(face[i + 1u], center)))) /
                                Real{6.0};
            const auto tetra_centroid =
                scale(add(add(center, face[0]), add(face[i], face[i + 1u])),
                      Real{0.25});
            moments.measure += volume;
            first_moment = add(first_moment, scale(tetra_centroid, volume));
        }
    }
    if (moments.measure <= Real{1.0e-30}) {
        return RegionMoments{};
    }
    moments.centroid = scale(first_moment, Real{1.0} / moments.measure);
    return moments;
}

[[nodiscard]] std::vector<std::vector<std::array<Real, 3>>> tetrahedronSideFaces(
    const std::vector<std::array<Real, 3>>& points,
    const std::vector<Real>& signed_values,
    const std::vector<CutPointCandidate>& ordered_cut_points,
    geometry::CutIntegrationSide side,
    Real tolerance)
{
    constexpr std::array<std::array<std::size_t, 3>, 4> faces{{
        {{0u, 1u, 2u}},
        {{0u, 1u, 3u}},
        {{0u, 2u, 3u}},
        {{1u, 2u, 3u}}}};
    std::vector<std::vector<std::array<Real, 3>>> clipped_faces;
    clipped_faces.reserve(5u);
    for (const auto& face : faces) {
        std::vector<SignedPoint> signed_face;
        signed_face.reserve(3u);
        for (const auto index : face) {
            signed_face.push_back(SignedPoint{points[index], signed_values[index]});
        }
        const auto clipped =
            side == geometry::CutIntegrationSide::Negative
                ? clipPolygonToNegativeLevelSet(signed_face, tolerance)
                : clipPolygonToPositiveLevelSet(signed_face, tolerance);
        if (clipped.size() >= 3u) {
            std::vector<std::array<Real, 3>> face_points;
            face_points.reserve(clipped.size());
            for (const auto& point : clipped) {
                face_points.push_back(point.point);
            }
            clipped_faces.push_back(std::move(face_points));
        }
    }
    if (ordered_cut_points.size() >= 3u) {
        std::vector<std::array<Real, 3>> cut_face;
        cut_face.reserve(ordered_cut_points.size());
        for (const auto& point : ordered_cut_points) {
            cut_face.push_back(point.point);
        }
        clipped_faces.push_back(std::move(cut_face));
    }
    return clipped_faces;
}

[[nodiscard]] std::vector<geometry::CutQuadraturePoint> polyhedronQuadratureFromFaces(
    const std::vector<std::vector<std::array<Real, 3>>>& faces,
    Real tolerance)
{
    std::vector<std::array<Real, 3>> unique_points;
    for (const auto& face : faces) {
        for (const auto& point : face) {
            addUniquePoint(unique_points, point, tolerance);
        }
    }
    if (unique_points.empty()) {
        return {};
    }

    std::array<Real, 3> center{{0.0, 0.0, 0.0}};
    for (const auto& point : unique_points) {
        center = add(center, point);
    }
    center = scale(center, Real{1.0} / static_cast<Real>(unique_points.size()));

    std::vector<geometry::CutQuadraturePoint> points;
    constexpr Real high = Real{0.5854101966249685};
    constexpr Real low = Real{0.1381966011250105};
    for (const auto& face : faces) {
        if (face.size() < 3u) {
            continue;
        }
        for (std::size_t i = 1u; i + 1u < face.size(); ++i) {
            const auto& a = center;
            const auto& b = face[0];
            const auto& c = face[i];
            const auto& d = face[i + 1u];
            const Real volume = tetraVolume(a, b, c, d);
            if (volume <= tolerance) {
                continue;
            }
            const Real weight = volume / Real{4.0};
            points.push_back(geometry::CutQuadraturePoint{
                .point = barycentricPoint(a, b, c, d, high, low, low, low),
                .weight = weight});
            points.push_back(geometry::CutQuadraturePoint{
                .point = barycentricPoint(a, b, c, d, low, high, low, low),
                .weight = weight});
            points.push_back(geometry::CutQuadraturePoint{
                .point = barycentricPoint(a, b, c, d, low, low, high, low),
                .weight = weight});
            points.push_back(geometry::CutQuadraturePoint{
                .point = barycentricPoint(a, b, c, d, low, low, low, high),
                .weight = weight});
        }
    }
    return points;
}

[[nodiscard]] RegionMoments complementMoments(
    const RegionMoments& parent,
    const RegionMoments& part)
{
    RegionMoments complement;
    complement.measure = parent.measure - part.measure;
    if (complement.measure <= Real{1.0e-30}) {
        complement.measure = Real{0.0};
        complement.centroid = parent.centroid;
        return complement;
    }
    const auto parent_moment = scale(parent.centroid, parent.measure);
    const auto part_moment = scale(part.centroid, part.measure);
    complement.centroid =
        scale(sub(parent_moment, part_moment), Real{1.0} / complement.measure);
    return complement;
}

[[nodiscard]] Real parentMeasure3D(
    const std::vector<std::array<Real, 3>>& points)
{
    return tetraVolume(points[0], points[1], points[2], points[3]);
}

[[nodiscard]] std::array<Real, 3> cellCentroid(
    const std::vector<std::array<Real, 3>>& points,
    std::size_t count) noexcept
{
    std::array<Real, 3> c{{0.0, 0.0, 0.0}};
    if (count == 0u) {
        return c;
    }
    for (std::size_t i = 0; i < count; ++i) {
        c = add(c, points[i]);
    }
    return scale(c, Real{1.0} / static_cast<Real>(count));
}

[[nodiscard]] RegionMoments parentMoments3D(
    const std::vector<std::array<Real, 3>>& points)
{
    RegionMoments moments;
    moments.measure = parentMeasure3D(points);
    moments.centroid = cellCentroid(points, 4u);
    return moments;
}

[[nodiscard]] CutInterfaceVolumeRegion makeVolumeRegion(
    const CutInterfaceDomainRequest& request,
    const LevelSetCellCutInput& input,
    geometry::CutIntegrationSide side,
    Real parent_measure,
    Real volume_fraction,
    const std::array<Real, 3>& centroid,
    const std::array<Real, 3>& interface_normal,
    const std::vector<Real>& signed_values,
    LocalIndex local_region_index,
    const std::string& suffix,
    std::vector<geometry::CutQuadraturePoint> quadrature_points = {},
    bool full_cell_equivalent = false)
{
    CutInterfaceVolumeRegion region;
    region.interface_marker = request.interface_marker;
    region.parent_cell = input.parent_cell;
    region.local_region_index = local_region_index;
    region.side = side;
    region.centroid = centroid;
    region.normal = side == geometry::CutIntegrationSide::Negative
                        ? interface_normal
                        : scale(interface_normal, Real{-1.0});
    region.parent_measure = parent_measure;
    region.volume_fraction = clampFraction(volume_fraction);
    region.measure = parent_measure * region.volume_fraction;
    region.min_level_set_value = *std::min_element(signed_values.begin(), signed_values.end());
    region.max_level_set_value = *std::max_element(signed_values.begin(), signed_values.end());
    region.topology_id = "cell-" + std::to_string(input.parent_cell) + "-" + suffix;
    region.full_cell_equivalent = full_cell_equivalent;
    for (auto& point : quadrature_points) {
        point.normal = region.normal;
    }
    region.quadrature_points = std::move(quadrature_points);
    region.stable_id = cutVolumeStableId(request.interface_marker,
                                         input.parent_cell,
                                         local_region_index,
                                         side,
                                         request.source.value_revision);
    return region;
}

void appendSideVolumeRegion(LevelSetCellCutResult& result,
                            CutInterfaceVolumeRegion region)
{
    if (region.measure > Real{0.0}) {
        result.volume_regions.push_back(std::move(region));
    }
}

} // namespace

bool supportsLinearLevelSetCellCut2D(ElementType element_type) noexcept
{
    switch (element_type) {
    case ElementType::Triangle3:
    case ElementType::Triangle6:
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
        return true;
    default:
        return false;
    }
}

bool supportsLinearLevelSetCellCut3D(ElementType element_type) noexcept
{
    switch (element_type) {
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return true;
    default:
        return false;
    }
}

bool isLevelSetCellCutExtensionElement(ElementType element_type) noexcept
{
    switch (element_type) {
    case ElementType::Hex8:
    case ElementType::Hex20:
    case ElementType::Hex27:
    case ElementType::Wedge6:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
    case ElementType::Pyramid5:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14:
        return true;
    default:
        return false;
    }
}

void LevelSetCellCutExtensionRegistry::registerCutter(
    LevelSetCellCutExtension extension)
{
    if (!isLevelSetCellCutExtensionElement(extension.element_type)) {
        throw std::invalid_argument("level-set cell cut extension requires hex, wedge, or pyramid element type");
    }
    if (extension.dimension != 3) {
        throw std::invalid_argument("level-set cell cut extension requires dimension 3");
    }
    if (extension.name.empty()) {
        throw std::invalid_argument("level-set cell cut extension requires a nonempty name");
    }
    if (!extension.cutter) {
        throw std::invalid_argument("level-set cell cut extension requires a cutter callback");
    }
    const auto key = static_cast<std::uint8_t>(extension.element_type);
    extensions_[key] = std::move(extension);
}

bool LevelSetCellCutExtensionRegistry::hasCutter(ElementType element_type) const noexcept
{
    const auto key = static_cast<std::uint8_t>(element_type);
    return extensions_.find(key) != extensions_.end();
}

std::vector<ElementType> LevelSetCellCutExtensionRegistry::registeredElementTypes() const
{
    std::vector<ElementType> types;
    types.reserve(extensions_.size());
    for (const auto& entry : extensions_) {
        types.push_back(entry.second.element_type);
    }
    std::sort(types.begin(),
              types.end(),
              [](ElementType a, ElementType b) {
                  return static_cast<std::uint8_t>(a) < static_cast<std::uint8_t>(b);
              });
    return types;
}

LevelSetCellCutResult LevelSetCellCutExtensionRegistry::cut(
    const CutInterfaceDomainRequest& request,
    const LevelSetCellCutInput& input) const
{
    const auto key = static_cast<std::uint8_t>(input.element_type);
    const auto it = extensions_.find(key);
    if (it == extensions_.end()) {
        LevelSetCellCutResult result;
        result.supported = false;
        result.degeneracy = CutInterfaceDegeneracy::NoCut;
        result.diagnostic = "no registered level-set cell cut extension for element type";
        return result;
    }
    return it->second.cutter(request, input);
}

LevelSetCellCutResult cutLinearLevelSetCell2D(const CutInterfaceDomainRequest& request,
                                              const LevelSetCellCutInput& input)
{
    LevelSetCellCutResult result;
    if (!supportsLinearLevelSetCellCut2D(input.element_type)) {
        result.supported = false;
        result.degeneracy = CutInterfaceDegeneracy::NoCut;
        result.diagnostic = "unsupported element type for linear 2D level-set cutting";
        return result;
    }
    if (!request.valid()) {
        throw std::invalid_argument("cutLinearLevelSetCell2D requires a valid interface request");
    }

    const std::size_t count = cornerCount(input.element_type);
    if (input.node_coordinates.size() < count || input.level_set_values.size() < count) {
        throw std::invalid_argument("linear 2D level-set cutting requires corner coordinates and values");
    }

    std::vector<Real> signed_values;
    signed_values.reserve(count);
    std::size_t zero_count = 0u;
    std::size_t negative_count = 0u;
    std::size_t positive_count = 0u;
    for (std::size_t i = 0; i < count; ++i) {
        const Real signed_value = input.level_set_values[i] - request.isovalue;
        signed_values.push_back(signed_value);
        if (std::abs(signed_value) <= request.tolerance) {
            ++zero_count;
        } else if (signed_value < Real{0.0}) {
            ++negative_count;
        } else {
            ++positive_count;
        }
    }

    const auto parent_moments = parentMoments2D(input.node_coordinates, count);
    const Real parent_measure = parent_moments.measure;
    const auto parent_centroid = parent_moments.centroid;
    const auto gradient_normal =
        estimateGradient2D(input.node_coordinates, signed_values, count);

    if (zero_count == count) {
        result.degeneracy = CutInterfaceDegeneracy::FullZeroCell;
        result.diagnostic = "all corner level-set values are on the requested isovalue";
        return result;
    }
    if (positive_count == 0u) {
        appendSideVolumeRegion(
            result,
            makeVolumeRegion(request,
                             input,
                             geometry::CutIntegrationSide::Negative,
                             parent_measure,
                             Real{1.0},
                             parent_centroid,
                             gradient_normal,
                             signed_values,
                             0u,
                             "full-negative-volume",
                             {},
                             true));
        result.degeneracy = classifyZeroContact2D(signed_values,
                                                  count,
                                                  request.tolerance);
        return result;
    }
    if (negative_count == 0u) {
        appendSideVolumeRegion(
            result,
            makeVolumeRegion(request,
                             input,
                             geometry::CutIntegrationSide::Positive,
                             parent_measure,
                             Real{1.0},
                             parent_centroid,
                             gradient_normal,
                             signed_values,
                             0u,
                             "full-positive-volume",
                             {},
                             true));
        result.degeneracy = classifyZeroContact2D(signed_values,
                                                  count,
                                                  request.tolerance);
        return result;
    }

    std::vector<CutPointCandidate> cut_points;
    cut_points.reserve(4u);
    bool zero_edge = false;
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t j = (i + 1u) % count;
        const Real di = signed_values[i];
        const Real dj = signed_values[j];
        const bool i_zero = std::abs(di) <= request.tolerance;
        const bool j_zero = std::abs(dj) <= request.tolerance;
        if (i_zero && j_zero) {
            zero_edge = true;
            addUniqueCandidate(
                cut_points,
                CutPointCandidate{input.node_coordinates[i], input.node_coordinates[i], 0.0},
                request.tolerance);
            addUniqueCandidate(
                cut_points,
                CutPointCandidate{input.node_coordinates[j], input.node_coordinates[j], 0.0},
                request.tolerance);
            continue;
        }
        if (i_zero) {
            addUniqueCandidate(
                cut_points,
                CutPointCandidate{input.node_coordinates[i], input.node_coordinates[i], 0.0},
                request.tolerance);
            continue;
        }
        if (j_zero) {
            addUniqueCandidate(
                cut_points,
                CutPointCandidate{input.node_coordinates[j], input.node_coordinates[j], 0.0},
                request.tolerance);
            continue;
        }
        if ((di < Real{0.0} && dj > Real{0.0}) ||
            (di > Real{0.0} && dj < Real{0.0})) {
            const Real t = di / (di - dj);
            const auto point = interpolate(input.node_coordinates[i], input.node_coordinates[j], t);
            addUniqueCandidate(cut_points,
                               CutPointCandidate{point, point, 0.0},
                               request.tolerance);
        }
    }

    if (cut_points.size() < 2u) {
        result.degeneracy = zero_count > 0u ? CutInterfaceDegeneracy::VertexTouch
                                            : CutInterfaceDegeneracy::SmallFragment;
        if (result.degeneracy == CutInterfaceDegeneracy::SmallFragment) {
            result.diagnostic = "level-set cut produced a fragment below the point separation tolerance";
        }
        return result;
    }

    const auto endpoints = farthestPair(cut_points);
    const auto& a = cut_points[endpoints.first];
    const auto& b = cut_points[endpoints.second];
    const Real measure = distance2(a.point, b.point);
    if (measure <= request.tolerance) {
        result.degeneracy = CutInterfaceDegeneracy::SmallFragment;
        result.diagnostic = "level-set cut produced a fragment below the minimum measure tolerance";
        return result;
    }

    const auto tangent = sub(b.point, a.point);
    std::array<Real, 3> normal{{tangent[1], -tangent[0], 0.0}};
    const Real normal_length = norm2(normal);
    if (normal_length > Real{1.0e-30}) {
        normal = scale(normal, Real{1.0} / normal_length);
        if (dot2(normal, gradient_normal) < Real{0.0}) {
            normal = scale(normal, Real{-1.0});
        }
    } else {
        normal = gradient_normal;
    }

    CutInterfaceFragment fragment;
    fragment.interface_marker = request.interface_marker;
    fragment.parent_cell = input.parent_cell;
    fragment.local_fragment_index = 0u;
    fragment.stable_id = cutInterfaceStableId(request.interface_marker,
                                              input.parent_cell,
                                              fragment.local_fragment_index,
                                              request.source.value_revision);
    fragment.kind = CutInterfaceFragmentKind::Segment;
    if (zero_edge || cut_points.size() > 2u) {
        fragment.degeneracy = CutInterfaceDegeneracy::EdgeTouch;
    } else if (zero_count > 0u) {
        fragment.degeneracy = CutInterfaceDegeneracy::VertexTouch;
    } else if (measure <= std::sqrt(request.tolerance)) {
        fragment.degeneracy = CutInterfaceDegeneracy::NearlyTangent;
    } else {
        fragment.degeneracy = CutInterfaceDegeneracy::None;
    }
    fragment.normal = normal;
    fragment.measure = measure;
    const auto negative_moments =
        cutSideMoments2D(input.node_coordinates,
                         signed_values,
                         count,
                         geometry::CutIntegrationSide::Negative,
                         request.tolerance);
    const auto positive_moments =
        cutSideMoments2D(input.node_coordinates,
                         signed_values,
                         count,
                         geometry::CutIntegrationSide::Positive,
                         request.tolerance);
    const auto negative_quadrature =
        cutSideQuadrature2D(input.node_coordinates,
                            signed_values,
                            count,
                            geometry::CutIntegrationSide::Negative,
                            request.tolerance);
    const auto positive_quadrature =
        cutSideQuadrature2D(input.node_coordinates,
                            signed_values,
                            count,
                            geometry::CutIntegrationSide::Positive,
                            request.tolerance);
    fragment.negative_volume_fraction =
        parent_measure > Real{0.0}
            ? clampFraction(negative_moments.measure / parent_measure)
            : Real{0.0};
    fragment.positive_volume_fraction = Real{1.0} - fragment.negative_volume_fraction;
    fragment.min_level_set_value = *std::min_element(signed_values.begin(), signed_values.end());
    fragment.max_level_set_value = *std::max_element(signed_values.begin(), signed_values.end());
    fragment.topology_id = "cell-" + std::to_string(input.parent_cell) + "-segment-0";
    fragment.vertices = {
        CutInterfaceVertex{.point = a.point,
                           .parent_coordinate = a.parent_coordinate,
                           .level_set_value = 0.0,
                           .stable_id = cutInterfaceStableId(request.interface_marker,
                                                             input.parent_cell,
                                                             0u,
                                                             request.source.value_revision)},
        CutInterfaceVertex{.point = b.point,
                           .parent_coordinate = b.parent_coordinate,
                           .level_set_value = 0.0,
                           .stable_id = cutInterfaceStableId(request.interface_marker,
                                                             input.parent_cell,
                                                             1u,
                                                             request.source.value_revision)}};
    fragment.quadrature_points = {
        CutInterfaceQuadraturePoint{.point = scale(add(a.point, b.point), Real{0.5}),
                                    .parent_coordinate = scale(add(a.parent_coordinate,
                                                                   b.parent_coordinate),
                                                               Real{0.5}),
                                    .normal = normal,
                                    .weight = measure}};

    result.degeneracy = fragment.degeneracy;
    appendSideVolumeRegion(
        result,
        makeVolumeRegion(request,
                         input,
                         geometry::CutIntegrationSide::Negative,
                         parent_measure,
                         fragment.negative_volume_fraction,
                         negative_moments.measure > Real{0.0}
                             ? negative_moments.centroid
                             : parent_centroid,
                         normal,
                         signed_values,
                         0u,
                         "cut-negative-volume",
                         negative_quadrature));
    appendSideVolumeRegion(
        result,
        makeVolumeRegion(request,
                         input,
                         geometry::CutIntegrationSide::Positive,
                         parent_measure,
                         fragment.positive_volume_fraction,
                         positive_moments.measure > Real{0.0}
                             ? positive_moments.centroid
                             : parent_centroid,
                         normal,
                         signed_values,
                         1u,
                         "cut-positive-volume",
                         positive_quadrature));
    result.fragments.push_back(std::move(fragment));
    return result;
}

LevelSetCellCutResult cutLinearLevelSetCell3D(const CutInterfaceDomainRequest& request,
                                              const LevelSetCellCutInput& input)
{
    LevelSetCellCutResult result;
    if (!supportsLinearLevelSetCellCut3D(input.element_type)) {
        result.supported = false;
        result.degeneracy = CutInterfaceDegeneracy::NoCut;
        result.diagnostic = "unsupported element type for linear 3D level-set cutting";
        return result;
    }
    if (!request.valid()) {
        throw std::invalid_argument("cutLinearLevelSetCell3D requires a valid interface request");
    }

    const std::size_t count = cornerCount3D(input.element_type);
    if (input.node_coordinates.size() < count || input.level_set_values.size() < count) {
        throw std::invalid_argument("linear 3D level-set cutting requires tetrahedron corner coordinates and values");
    }

    std::vector<Real> signed_values;
    signed_values.reserve(count);
    std::size_t zero_count = 0u;
    std::size_t negative_count = 0u;
    std::size_t positive_count = 0u;
    for (std::size_t i = 0; i < count; ++i) {
        const Real signed_value = input.level_set_values[i] - request.isovalue;
        signed_values.push_back(signed_value);
        if (std::abs(signed_value) <= request.tolerance) {
            ++zero_count;
        } else if (signed_value < Real{0.0}) {
            ++negative_count;
        } else {
            ++positive_count;
        }
    }

    const auto parent_moments = parentMoments3D(input.node_coordinates);
    const Real parent_measure = parent_moments.measure;
    const auto parent_centroid = parent_moments.centroid;
    const auto gradient_normal =
        estimateGradient3D(input.node_coordinates, signed_values, count);

    if (zero_count == count) {
        result.degeneracy = CutInterfaceDegeneracy::FullZeroCell;
        result.diagnostic = "all tetrahedron corner level-set values are on the requested isovalue";
        return result;
    }
    if (positive_count == 0u) {
        appendSideVolumeRegion(
            result,
            makeVolumeRegion(request,
                             input,
                             geometry::CutIntegrationSide::Negative,
                             parent_measure,
                             Real{1.0},
                             parent_centroid,
                             gradient_normal,
                             signed_values,
                             0u,
                             "full-negative-volume",
                             {},
                             true));
        result.degeneracy =
            classifyZeroContactTetrahedron(signed_values, request.tolerance);
        return result;
    }
    if (negative_count == 0u) {
        appendSideVolumeRegion(
            result,
            makeVolumeRegion(request,
                             input,
                             geometry::CutIntegrationSide::Positive,
                             parent_measure,
                             Real{1.0},
                             parent_centroid,
                             gradient_normal,
                             signed_values,
                             0u,
                             "full-positive-volume",
                             {},
                             true));
        result.degeneracy =
            classifyZeroContactTetrahedron(signed_values, request.tolerance);
        return result;
    }

    constexpr std::array<std::array<std::size_t, 2>, 6> edges{{
        {{0u, 1u}},
        {{0u, 2u}},
        {{0u, 3u}},
        {{1u, 2u}},
        {{1u, 3u}},
        {{2u, 3u}}}};

    std::vector<CutPointCandidate> cut_points;
    cut_points.reserve(4u);
    bool zero_edge = false;
    for (const auto& edge : edges) {
        const std::size_t i = edge[0];
        const std::size_t j = edge[1];
        const Real di = signed_values[i];
        const Real dj = signed_values[j];
        const bool i_zero = std::abs(di) <= request.tolerance;
        const bool j_zero = std::abs(dj) <= request.tolerance;
        if (i_zero && j_zero) {
            zero_edge = true;
            addUniqueCandidate(
                cut_points,
                CutPointCandidate{input.node_coordinates[i], input.node_coordinates[i], 0.0},
                request.tolerance);
            addUniqueCandidate(
                cut_points,
                CutPointCandidate{input.node_coordinates[j], input.node_coordinates[j], 0.0},
                request.tolerance);
            continue;
        }
        if (i_zero) {
            addUniqueCandidate(
                cut_points,
                CutPointCandidate{input.node_coordinates[i], input.node_coordinates[i], 0.0},
                request.tolerance);
            continue;
        }
        if (j_zero) {
            addUniqueCandidate(
                cut_points,
                CutPointCandidate{input.node_coordinates[j], input.node_coordinates[j], 0.0},
                request.tolerance);
            continue;
        }
        if ((di < Real{0.0} && dj > Real{0.0}) ||
            (di > Real{0.0} && dj < Real{0.0})) {
            const Real t = di / (di - dj);
            const auto point = interpolate(input.node_coordinates[i], input.node_coordinates[j], t);
            addUniqueCandidate(cut_points,
                               CutPointCandidate{point, point, 0.0},
                               request.tolerance);
        }
    }

    if (cut_points.size() < 3u) {
        if (zero_count > 0u) {
            result.degeneracy = cut_points.size() == 2u ? CutInterfaceDegeneracy::EdgeTouch
                                                        : CutInterfaceDegeneracy::VertexTouch;
        } else {
            result.degeneracy = CutInterfaceDegeneracy::SmallFragment;
            result.diagnostic = "level-set cut produced a fragment below the point separation tolerance";
        }
        return result;
    }

    orderPolygonPoints(cut_points, gradient_normal);
    const Real measure = polygonArea(cut_points, gradient_normal);
    if (measure <= request.tolerance) {
        result.degeneracy = CutInterfaceDegeneracy::SmallFragment;
        result.diagnostic = "level-set cut produced a fragment below the minimum measure tolerance";
        return result;
    }

    auto normal = unitOrDefault(
        cross(sub(cut_points[1].point, cut_points[0].point),
              sub(cut_points[2].point, cut_points[0].point)),
        gradient_normal);
    if (dot3(normal, gradient_normal) < Real{0.0}) {
        normal = scale(normal, Real{-1.0});
    }

    CutInterfaceFragment fragment;
    fragment.interface_marker = request.interface_marker;
    fragment.parent_cell = input.parent_cell;
    fragment.local_fragment_index = 0u;
    fragment.stable_id = cutInterfaceStableId(request.interface_marker,
                                              input.parent_cell,
                                              fragment.local_fragment_index,
                                              request.source.value_revision);
    fragment.kind = CutInterfaceFragmentKind::Polygon;
    if (zero_edge || cut_points.size() > 4u) {
        fragment.degeneracy = CutInterfaceDegeneracy::EdgeTouch;
    } else if (zero_count > 0u) {
        fragment.degeneracy = CutInterfaceDegeneracy::VertexTouch;
    } else if (measure <= std::sqrt(request.tolerance)) {
        fragment.degeneracy = CutInterfaceDegeneracy::NearlyTangent;
    } else {
        fragment.degeneracy = CutInterfaceDegeneracy::None;
    }
    fragment.normal = normal;
    fragment.measure = measure;
    const auto negative_faces =
        tetrahedronSideFaces(input.node_coordinates,
                             signed_values,
                             cut_points,
                             geometry::CutIntegrationSide::Negative,
                             request.tolerance);
    const auto positive_faces =
        tetrahedronSideFaces(input.node_coordinates,
                             signed_values,
                             cut_points,
                             geometry::CutIntegrationSide::Positive,
                             request.tolerance);
    const auto negative_moments =
        polyhedronMomentsFromFaces(negative_faces, request.tolerance);
    auto positive_moments =
        polyhedronMomentsFromFaces(positive_faces, request.tolerance);
    if (positive_moments.measure <= Real{1.0e-30}) {
        positive_moments = complementMoments(parent_moments, negative_moments);
    }
    const auto negative_quadrature =
        polyhedronQuadratureFromFaces(negative_faces, request.tolerance);
    const auto positive_quadrature =
        polyhedronQuadratureFromFaces(positive_faces, request.tolerance);
    fragment.negative_volume_fraction =
        parent_measure > Real{0.0}
            ? clampFraction(negative_moments.measure / parent_measure)
            : Real{0.0};
    fragment.positive_volume_fraction = Real{1.0} - fragment.negative_volume_fraction;
    fragment.min_level_set_value = *std::min_element(signed_values.begin(), signed_values.end());
    fragment.max_level_set_value = *std::max_element(signed_values.begin(), signed_values.end());
    fragment.topology_id = "cell-" + std::to_string(input.parent_cell) + "-polygon-0";
    fragment.vertices.reserve(cut_points.size());
    for (std::size_t i = 0; i < cut_points.size(); ++i) {
        const auto stable_index = static_cast<LocalIndex>(i + 1u);
        fragment.vertices.push_back(
            CutInterfaceVertex{.point = cut_points[i].point,
                               .parent_coordinate = cut_points[i].parent_coordinate,
                               .level_set_value = 0.0,
                               .stable_id = cutInterfaceStableId(request.interface_marker,
                                                                 input.parent_cell,
                                                                 stable_index,
                                                                 request.source.value_revision)});
    }
    const auto qp = centroid(cut_points);
    fragment.quadrature_points = {
        CutInterfaceQuadraturePoint{.point = qp,
                                    .parent_coordinate = qp,
                                    .normal = normal,
                                    .weight = measure}};

    result.degeneracy = fragment.degeneracy;
    appendSideVolumeRegion(
        result,
        makeVolumeRegion(request,
                         input,
                         geometry::CutIntegrationSide::Negative,
                         parent_measure,
                         fragment.negative_volume_fraction,
                         negative_moments.measure > Real{0.0}
                             ? negative_moments.centroid
                             : parent_centroid,
                         normal,
                         signed_values,
                         0u,
                         "cut-negative-volume",
                         negative_quadrature));
    appendSideVolumeRegion(
        result,
        makeVolumeRegion(request,
                         input,
                         geometry::CutIntegrationSide::Positive,
                         parent_measure,
                         fragment.positive_volume_fraction,
                         positive_moments.measure > Real{0.0}
                             ? positive_moments.centroid
                             : parent_centroid,
                         normal,
                         signed_values,
                         1u,
                         "cut-positive-volume",
                         positive_quadrature));
    result.fragments.push_back(std::move(fragment));
    return result;
}

void appendLinearLevelSetCellCut2D(LevelSetInterfaceDomain& domain,
                                   const LevelSetCellCutInput& input)
{
    auto result = cutLinearLevelSetCell2D(domain.request(), input);
    for (auto& region : result.volume_regions) {
        domain.addVolumeRegion(std::move(region));
    }
    for (auto& fragment : result.fragments) {
        domain.addFragment(std::move(fragment));
    }
}

void appendLinearLevelSetCellCut3D(LevelSetInterfaceDomain& domain,
                                   const LevelSetCellCutInput& input)
{
    auto result = cutLinearLevelSetCell3D(domain.request(), input);
    for (auto& region : result.volume_regions) {
        domain.addVolumeRegion(std::move(region));
    }
    for (auto& fragment : result.fragments) {
        domain.addFragment(std::move(fragment));
    }
}

} // namespace interfaces
} // namespace FE
} // namespace svmp
