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

    if (zero_count == count) {
        result.degeneracy = CutInterfaceDegeneracy::FullZeroCell;
        result.diagnostic = "all corner level-set values are on the requested isovalue";
        return result;
    }
    if (positive_count == 0u && zero_count == 0u) {
        result.degeneracy = CutInterfaceDegeneracy::NoCut;
        return result;
    }
    if (negative_count == 0u && zero_count == 0u) {
        result.degeneracy = CutInterfaceDegeneracy::NoCut;
        return result;
    }

    std::vector<CutPointCandidate> cut_points;
    cut_points.reserve(4u);
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t j = (i + 1u) % count;
        const Real di = signed_values[i];
        const Real dj = signed_values[j];
        const bool i_zero = std::abs(di) <= request.tolerance;
        const bool j_zero = std::abs(dj) <= request.tolerance;
        if (i_zero && j_zero) {
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
                                            : CutInterfaceDegeneracy::NoCut;
        return result;
    }

    const auto endpoints = farthestPair(cut_points);
    const auto& a = cut_points[endpoints.first];
    const auto& b = cut_points[endpoints.second];
    const Real measure = distance2(a.point, b.point);
    if (measure <= request.tolerance) {
        result.degeneracy = CutInterfaceDegeneracy::SmallFragment;
        return result;
    }

    const auto gradient_normal =
        estimateGradient2D(input.node_coordinates, signed_values, count);
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
    fragment.degeneracy =
        cut_points.size() > 2u ? CutInterfaceDegeneracy::EdgeTouch
                               : CutInterfaceDegeneracy::None;
    fragment.normal = normal;
    fragment.measure = measure;
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

    if (zero_count == count) {
        result.degeneracy = CutInterfaceDegeneracy::FullZeroCell;
        result.diagnostic = "all tetrahedron corner level-set values are on the requested isovalue";
        return result;
    }
    if (positive_count == 0u && zero_count == 0u) {
        result.degeneracy = CutInterfaceDegeneracy::NoCut;
        return result;
    }
    if (negative_count == 0u && zero_count == 0u) {
        result.degeneracy = CutInterfaceDegeneracy::NoCut;
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
    for (const auto& edge : edges) {
        const std::size_t i = edge[0];
        const std::size_t j = edge[1];
        const Real di = signed_values[i];
        const Real dj = signed_values[j];
        const bool i_zero = std::abs(di) <= request.tolerance;
        const bool j_zero = std::abs(dj) <= request.tolerance;
        if (i_zero && j_zero) {
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
        result.degeneracy = cut_points.size() == 2u ? CutInterfaceDegeneracy::EdgeTouch
                                                    : CutInterfaceDegeneracy::VertexTouch;
        return result;
    }

    const auto gradient_normal =
        estimateGradient3D(input.node_coordinates, signed_values, count);
    orderPolygonPoints(cut_points, gradient_normal);
    const Real measure = polygonArea(cut_points, gradient_normal);
    if (measure <= request.tolerance) {
        result.degeneracy = CutInterfaceDegeneracy::SmallFragment;
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
    fragment.degeneracy =
        cut_points.size() > 4u ? CutInterfaceDegeneracy::EdgeTouch
                               : CutInterfaceDegeneracy::None;
    fragment.normal = normal;
    fragment.measure = measure;
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
    result.fragments.push_back(std::move(fragment));
    return result;
}

void appendLinearLevelSetCellCut2D(LevelSetInterfaceDomain& domain,
                                   const LevelSetCellCutInput& input)
{
    auto result = cutLinearLevelSetCell2D(domain.request(), input);
    for (auto& fragment : result.fragments) {
        domain.addFragment(std::move(fragment));
    }
}

void appendLinearLevelSetCellCut3D(LevelSetInterfaceDomain& domain,
                                   const LevelSetCellCutInput& input)
{
    auto result = cutLinearLevelSetCell3D(domain.request(), input);
    for (auto& fragment : result.fragments) {
        domain.addFragment(std::move(fragment));
    }
}

} // namespace interfaces
} // namespace FE
} // namespace svmp
