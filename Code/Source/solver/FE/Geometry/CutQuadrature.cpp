/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Geometry/CutQuadrature.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace svmp {
namespace FE {
namespace geometry {
namespace {

Real dot(const std::array<Real, 3>& a, const std::array<Real, 3>& b) noexcept {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<Real, 3> sub(const std::array<Real, 3>& a,
                        const std::array<Real, 3>& b) noexcept {
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

std::array<Real, 3> add(const std::array<Real, 3>& a,
                        const std::array<Real, 3>& b) noexcept {
    return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
}

std::array<Real, 3> scale(const std::array<Real, 3>& a, Real s) noexcept {
    return {{a[0] * s, a[1] * s, a[2] * s}};
}

Real norm(const std::array<Real, 3>& a) noexcept {
    return std::sqrt(dot(a, a));
}

std::array<Real, 3> cross(const std::array<Real, 3>& a,
                          const std::array<Real, 3>& b) noexcept {
    return {{a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]}};
}

std::array<Real, 3> unit_or_default(std::array<Real, 3> n) noexcept {
    const Real len = norm(n);
    if (len <= Real{1.0e-30}) {
        return {{1.0, 0.0, 0.0}};
    }
    return scale(n, Real{1.0} / len);
}

Real box_measure(const std::array<Real, 3>& min_corner,
                 const std::array<Real, 3>& max_corner) {
    return std::max(Real{0.0}, max_corner[0] - min_corner[0]) *
           std::max(Real{0.0}, max_corner[1] - min_corner[1]) *
           std::max(Real{0.0}, max_corner[2] - min_corner[2]);
}

void validate_axis(int axis) {
    if (axis < 0 || axis > 2) {
        throw std::invalid_argument("cut quadrature axis must be 0, 1, or 2");
    }
}

Real polygon_area_projected(const std::vector<std::array<Real, 3>>& points,
                            const std::array<Real, 3>& normal) noexcept {
    if (points.size() < 3u) {
        return Real{0.0};
    }
    std::array<Real, 3> area_vec{{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < points.size(); ++i) {
        area_vec = add(area_vec, cross(points[i], points[(i + 1u) % points.size()]));
    }
    return Real{0.5} * std::abs(dot(area_vec, unit_or_default(normal)));
}

Real interface_measure(const std::vector<std::array<Real, 3>>& points,
                       const std::array<Real, 3>& normal) noexcept {
    if (points.empty()) {
        return Real{0.0};
    }
    if (points.size() == 1u) {
        return Real{1.0};
    }
    if (points.size() == 2u) {
        return norm(sub(points[1], points[0]));
    }
    const Real area = polygon_area_projected(points, normal);
    if (area > Real{1.0e-30}) {
        return area;
    }
    Real length = Real{0.0};
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (std::size_t j = i + 1u; j < points.size(); ++j) {
            length = std::max(length, norm(sub(points[j], points[i])));
        }
    }
    return length;
}

std::array<Real, 3> centroid(const std::array<Real, 3>& a,
                             const std::array<Real, 3>& b) noexcept {
    return scale(add(a, b), Real{0.5});
}

std::array<Real, 3> centroid(const std::vector<std::array<Real, 3>>& points) noexcept {
    std::array<Real, 3> c{{0.0, 0.0, 0.0}};
    if (points.empty()) {
        return c;
    }
    for (const auto& p : points) {
        c = add(c, p);
    }
    return scale(c, Real{1.0} / static_cast<Real>(points.size()));
}

void attach_policy(CutQuadratureRule& rule,
                   CutQuadratureConstructionKind kind,
                   std::string provenance_id) {
    rule.policy.kind = kind;
    rule.provenance.construction = kind;
    rule.provenance_id = std::move(provenance_id);
    rule.provenance.embedded_geometry_id = rule.provenance_id;
    rule.frame = rule.provenance.frame;
}

} // namespace

CutQuadratureRule makeAxisAlignedBoxCutVolumeQuadrature(
    const std::array<Real, 3>& min_corner,
    const std::array<Real, 3>& max_corner,
    int axis,
    Real cut_coordinate,
    CutIntegrationSide side,
    std::string provenance_id) {
    validate_axis(axis);
    if (side == CutIntegrationSide::Interface) {
        throw std::invalid_argument("volume cut quadrature requires Negative or Positive side");
    }

    const Real parent = box_measure(min_corner, max_corner);
    std::array<Real, 3> clipped_min = min_corner;
    std::array<Real, 3> clipped_max = max_corner;
    if (side == CutIntegrationSide::Negative) {
        clipped_max[static_cast<std::size_t>(axis)] =
            std::min(max_corner[static_cast<std::size_t>(axis)], cut_coordinate);
    } else {
        clipped_min[static_cast<std::size_t>(axis)] =
            std::max(min_corner[static_cast<std::size_t>(axis)], cut_coordinate);
    }
    const Real measure = box_measure(clipped_min, clipped_max);

    CutQuadratureRule rule;
    rule.kind = CutQuadratureKind::Volume;
    rule.side = side;
    rule.measure = measure;
    rule.parent_measure = parent;
    rule.volume_fraction = parent > Real{0.0} ? measure / parent : Real{0.0};
    attach_policy(rule, CutQuadratureConstructionKind::AxisAlignedBoxClip, std::move(provenance_id));
    if (measure > Real{0.0}) {
        CutQuadraturePoint qp;
        qp.weight = measure;
        for (int d = 0; d < 3; ++d) {
            qp.point[static_cast<std::size_t>(d)] =
                Real{0.5} * (clipped_min[static_cast<std::size_t>(d)] +
                             clipped_max[static_cast<std::size_t>(d)]);
        }
        qp.normal[static_cast<std::size_t>(axis)] =
            side == CutIntegrationSide::Negative ? Real{1.0} : Real{-1.0};
        rule.points.push_back(qp);
    }
    return rule;
}

CutQuadratureRule makeAxisAlignedBoxCutInterfaceQuadrature(
    const std::array<Real, 3>& min_corner,
    const std::array<Real, 3>& max_corner,
    int axis,
    Real cut_coordinate,
    std::string provenance_id) {
    validate_axis(axis);
    const auto ax = static_cast<std::size_t>(axis);
    const Real parent = box_measure(min_corner, max_corner);
    const bool inside = cut_coordinate >= min_corner[ax] && cut_coordinate <= max_corner[ax];

    CutQuadratureRule rule;
    rule.kind = CutQuadratureKind::Interface;
    rule.side = CutIntegrationSide::Interface;
    rule.parent_measure = parent;
    attach_policy(rule, CutQuadratureConstructionKind::AxisAlignedBoxClip, std::move(provenance_id));
    if (!inside || parent <= Real{0.0}) {
        return rule;
    }

    int a0 = (axis + 1) % 3;
    int a1 = (axis + 2) % 3;
    rule.measure = (max_corner[static_cast<std::size_t>(a0)] - min_corner[static_cast<std::size_t>(a0)]) *
                   (max_corner[static_cast<std::size_t>(a1)] - min_corner[static_cast<std::size_t>(a1)]);
    rule.volume_fraction = Real{0.0};
    CutQuadraturePoint qp;
    qp.weight = rule.measure;
    qp.point = {{Real{0.5} * (min_corner[0] + max_corner[0]),
                 Real{0.5} * (min_corner[1] + max_corner[1]),
                 Real{0.5} * (min_corner[2] + max_corner[2])}};
    qp.point[ax] = cut_coordinate;
    qp.normal[ax] = Real{1.0};
    rule.points.push_back(qp);
    return rule;
}

CutQuadratureRule makeSegmentCutFaceQuadrature(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b,
    const std::array<Real, 3>& plane_origin,
    const std::array<Real, 3>& plane_normal,
    CutIntegrationSide side,
    std::string provenance_id) {
    if (side == CutIntegrationSide::Interface) {
        throw std::invalid_argument("segment cut-face quadrature requires Negative or Positive side");
    }
    const auto n = unit_or_default(plane_normal);
    const Real da = dot(sub(a, plane_origin), n);
    const Real db = dot(sub(b, plane_origin), n);
    const Real parent = norm(sub(b, a));
    const CutQuadratureConstructionPolicy default_policy;
    const Real tol = default_policy.tolerance;
    const auto on_side = [&](Real d) {
        return side == CutIntegrationSide::Negative ? d <= tol : d >= -tol;
    };

    std::array<Real, 3> seg_min = a;
    std::array<Real, 3> seg_max = a;
    if (parent > Real{0.0}) {
        const bool a_on = on_side(da);
        const bool b_on = on_side(db);
        if (a_on && b_on) {
            seg_min = a;
            seg_max = b;
        } else if (a_on || b_on) {
            Real t = Real{0.0};
            if (std::abs(da - db) > Real{1.0e-30}) {
                t = da / (da - db);
            }
            t = std::max(Real{0.0}, std::min(Real{1.0}, t));
            const auto hit = add(scale(a, Real{1.0} - t), scale(b, t));
            if (a_on) {
                seg_min = a;
                seg_max = hit;
            } else {
                seg_min = hit;
                seg_max = b;
            }
        }
    }

    const Real measure = norm(sub(seg_max, seg_min));
    CutQuadratureRule rule;
    rule.kind = CutQuadratureKind::Face;
    rule.side = side;
    rule.measure = measure;
    rule.parent_measure = parent;
    rule.volume_fraction = parent > Real{0.0} ? measure / parent : Real{0.0};
    attach_policy(rule, CutQuadratureConstructionKind::SegmentClip, std::move(provenance_id));
    if (measure > Real{0.0}) {
        CutQuadraturePoint qp;
        qp.weight = measure;
        qp.point = centroid(seg_min, seg_max);
        qp.normal = side == CutIntegrationSide::Negative ? n : scale(n, Real{-1.0});
        rule.points.push_back(qp);
    }
    return rule;
}

CutQuadratureRule makePolygonInterfaceQuadrature(
    const std::vector<std::array<Real, 3>>& ordered_points,
    const std::array<Real, 3>& normal,
    const CutQuadratureConstructionPolicy& policy,
    CutQuadratureProvenance provenance) {
    CutQuadratureRule rule;
    rule.kind = CutQuadratureKind::Interface;
    rule.side = CutIntegrationSide::Interface;
    rule.policy = policy;
    rule.policy.kind = CutQuadratureConstructionKind::PolygonInterface;
    rule.provenance = std::move(provenance);
    rule.provenance.construction = CutQuadratureConstructionKind::PolygonInterface;
    rule.provenance_id = rule.provenance.embedded_geometry_id;
    rule.frame = rule.provenance.frame;
    rule.measure = interface_measure(ordered_points, normal);
    rule.parent_measure = rule.measure;
    rule.volume_fraction = Real{0.0};
    rule.exact_for_constants = true;
    rule.exact_polynomial_order = std::max(0, policy.polynomial_order);
    if (rule.measure > Real{0.0}) {
        CutQuadraturePoint qp;
        qp.weight = rule.measure;
        qp.point = centroid(ordered_points);
        qp.normal = unit_or_default(normal);
        rule.points.push_back(qp);
    }
    return rule;
}

CutTopologyQuadratureRules makeTopologyDerivedCutQuadrature(
    const CutTopologyQuadratureInput& input) {
    CutTopologyQuadratureRules out;
    auto policy = input.policy;
    policy.kind = CutQuadratureConstructionKind::TopologySubdivision;
    policy.moment_fitted = false;
    auto provenance = input.provenance;
    provenance.construction = CutQuadratureConstructionKind::TopologySubdivision;
    provenance.frame = input.frame;

    const auto volumes = makeConservativeSplitCutVolumeQuadrature(
        input.parent_measure,
        input.negative_fraction,
        input.representative_point,
        input.interface_normal,
        policy,
        provenance);
    out.negative_volume = volumes[0];
    out.positive_volume = volumes[1];
    out.negative_volume.policy.kind = CutQuadratureConstructionKind::TopologySubdivision;
    out.positive_volume.policy.kind = CutQuadratureConstructionKind::TopologySubdivision;
    out.negative_volume.provenance.construction = CutQuadratureConstructionKind::TopologySubdivision;
    out.positive_volume.provenance.construction = CutQuadratureConstructionKind::TopologySubdivision;
    out.negative_volume.frame = input.frame;
    out.positive_volume.frame = input.frame;
    out.negative_volume.curved_geometry = input.curved_geometry;
    out.positive_volume.curved_geometry = input.curved_geometry;

    out.interface_rule = makePolygonInterfaceQuadrature(
        input.ordered_interface_points,
        input.interface_normal,
        policy,
        provenance);
    out.interface_rule.policy.kind = CutQuadratureConstructionKind::TopologySubdivision;
    out.interface_rule.provenance.construction = CutQuadratureConstructionKind::TopologySubdivision;
    out.interface_rule.frame = input.frame;
    out.interface_rule.curved_geometry = input.curved_geometry;
    out.conservation = checkCutVolumeConservation(out.negative_volume, out.positive_volume, policy.tolerance);
    out.diagnostic = diagnoseCutQuadrature(out.negative_volume);
    if (!out.conservation.ok) {
        out.diagnostic.ok = false;
        out.diagnostic.conservation_failure = true;
        out.diagnostic.messages.push_back("topology-derived split volume quadrature is not conservative");
    }
    return out;
}

CutTopologyQuadratureRules makeClosedTopologyCutQuadrature(
    const CutClosedTopologyQuadratureInput& input) {
    CutTopologyQuadratureRules out;
    const auto has_curved_subcells = [](const std::vector<CutTopologySubcellInput>& subcells) {
        return std::any_of(subcells.begin(), subcells.end(), [](const auto& subcell) {
            return subcell.curved_geometry || subcell.measure_from_isoparametric_quadrature;
        });
    };
    const bool curved_topology =
        input.curved_geometry ||
        !input.curved_interface_points.empty() ||
        has_curved_subcells(input.negative_subcells) ||
        has_curved_subcells(input.positive_subcells);
    const auto construction_kind = curved_topology
                                       ? CutQuadratureConstructionKind::CurvedTopologySubdivision
                                       : CutQuadratureConstructionKind::TopologySubdivision;
    auto policy = input.policy;
    policy.kind = construction_kind;
    policy.moment_fitted = false;
    if (curved_topology && policy.name == "constant-exact") {
        policy.name = "curved-isoparametric-topology-subdivision";
    }
    auto provenance = input.provenance;
    provenance.construction = construction_kind;
    provenance.frame = input.frame;

    const auto make_rule = [&](CutIntegrationSide side,
                               const std::vector<CutTopologySubcellInput>& subcells) {
        CutQuadratureRule rule;
        rule.kind = CutQuadratureKind::Volume;
        rule.side = side;
        rule.parent_measure = std::max(Real{0.0}, input.parent_measure);
        rule.exact_for_constants = true;
        rule.exact_polynomial_order = std::max(0, policy.polynomial_order);
        rule.policy = policy;
        rule.provenance = provenance;
        rule.provenance_id = provenance.embedded_geometry_id;
        rule.frame = input.frame;
        rule.curved_geometry = curved_topology;
        for (const auto& subcell : subcells) {
            if (subcell.measure <= Real{0.0}) {
                continue;
            }
            CutQuadraturePoint qp;
            qp.weight = subcell.measure;
            const bool finite_centroid = std::isfinite(subcell.centroid[0]) &&
                                         std::isfinite(subcell.centroid[1]) &&
                                         std::isfinite(subcell.centroid[2]);
            qp.point = finite_centroid ? subcell.centroid : centroid(subcell.points);
            qp.normal = side == CutIntegrationSide::Negative
                            ? unit_or_default(input.interface_normal)
                            : scale(unit_or_default(input.interface_normal), Real{-1.0});
            rule.curved_geometry = rule.curved_geometry ||
                                   subcell.curved_geometry ||
                                   subcell.measure_from_isoparametric_quadrature;
            rule.measure += subcell.measure;
            rule.points.push_back(qp);
        }
        rule.volume_fraction =
            rule.parent_measure > Real{0.0} ? rule.measure / rule.parent_measure : Real{0.0};
        return rule;
    };

    out.negative_volume = make_rule(CutIntegrationSide::Negative, input.negative_subcells);
    out.positive_volume = make_rule(CutIntegrationSide::Positive, input.positive_subcells);
    if (!input.curved_interface_points.empty()) {
        out.interface_rule = makeCurvedInterfaceQuadrature(
            input.curved_interface_points,
            input.frame,
            policy,
            provenance);
        out.interface_rule.policy.kind = construction_kind;
        out.interface_rule.provenance.construction = construction_kind;
    } else {
        out.interface_rule = makePolygonInterfaceQuadrature(
            input.ordered_interface_points,
            input.interface_normal,
            policy,
            provenance);
        out.interface_rule.policy.kind = construction_kind;
        out.interface_rule.provenance.construction = construction_kind;
        out.interface_rule.curved_geometry = curved_topology;
    }
    out.interface_rule.frame = input.frame;
    out.interface_rule.curved_geometry = out.interface_rule.curved_geometry || curved_topology;
    out.conservation = checkCutVolumeConservation(out.negative_volume, out.positive_volume, policy.tolerance);
    out.diagnostic = diagnoseCutQuadrature(out.negative_volume);
    const auto positive_diagnostic = diagnoseCutQuadrature(out.positive_volume);
    if (!positive_diagnostic.ok) {
        out.diagnostic.ok = false;
        out.diagnostic.messages.insert(out.diagnostic.messages.end(),
                                       positive_diagnostic.messages.begin(),
                                       positive_diagnostic.messages.end());
    }
    if (!out.conservation.ok) {
        out.diagnostic.ok = false;
        out.diagnostic.conservation_failure = true;
        out.diagnostic.messages.push_back("closed-topology cut quadrature is not conservative");
    }
    return out;
}

CutQuadratureRule makeCurvedInterfaceQuadrature(
    const std::vector<CutQuadraturePoint>& points,
    CutGeometryFrame frame,
    const CutQuadratureConstructionPolicy& policy,
    CutQuadratureProvenance provenance) {
    CutQuadratureRule rule;
    rule.kind = CutQuadratureKind::Interface;
    rule.side = CutIntegrationSide::Interface;
    rule.policy = policy;
    rule.policy.kind = CutQuadratureConstructionKind::CurvedInterface;
    rule.provenance = std::move(provenance);
    rule.provenance.construction = CutQuadratureConstructionKind::CurvedInterface;
    rule.provenance.frame = frame;
    rule.provenance_id = rule.provenance.embedded_geometry_id;
    rule.frame = frame;
    rule.curved_geometry = true;
    rule.exact_for_constants = true;
    rule.exact_polynomial_order = std::max(0, policy.polynomial_order);
    for (const auto& qp : points) {
        if (qp.weight <= Real{0.0}) {
            continue;
        }
        CutQuadraturePoint normalized = qp;
        normalized.normal = unit_or_default(qp.normal);
        rule.measure += qp.weight;
        rule.points.push_back(normalized);
    }
    rule.parent_measure = rule.measure;
    return rule;
}

CutQuadratureRule makeMomentFittedCutVolumeQuadrature(
    Real parent_measure,
    Real side_fraction,
    CutIntegrationSide side,
    const std::array<Real, 3>& moment_point,
    const std::array<Real, 3>& interface_normal,
    int exact_polynomial_order,
    CutQuadratureProvenance provenance) {
    if (side == CutIntegrationSide::Interface) {
        throw std::invalid_argument("moment-fitted volume quadrature requires Negative or Positive side");
    }
    CutQuadratureConstructionPolicy policy;
    policy.kind = CutQuadratureConstructionKind::MomentFittedImplicit;
    policy.moment_fitted = true;
    policy.polynomial_order = std::max(0, exact_polynomial_order);
    policy.name = "moment-fitted-implicit";
    side_fraction = std::max(Real{0.0}, std::min(Real{1.0}, side_fraction));

    CutQuadratureRule rule;
    rule.kind = CutQuadratureKind::Volume;
    rule.side = side;
    rule.parent_measure = std::max(Real{0.0}, parent_measure);
    rule.volume_fraction = side_fraction;
    rule.measure = rule.parent_measure * side_fraction;
    rule.policy = policy;
    rule.provenance = std::move(provenance);
    rule.provenance.construction = CutQuadratureConstructionKind::MomentFittedImplicit;
    rule.provenance_id = rule.provenance.embedded_geometry_id;
    rule.frame = rule.provenance.frame;
    rule.exact_for_constants = true;
    rule.exact_polynomial_order = policy.polynomial_order;
    if (rule.measure > Real{0.0}) {
        CutQuadraturePoint qp;
        qp.point = moment_point;
        qp.weight = rule.measure;
        qp.normal = side == CutIntegrationSide::Negative
                        ? unit_or_default(interface_normal)
                        : scale(unit_or_default(interface_normal), Real{-1.0});
        rule.points.push_back(qp);
    }
    return rule;
}

std::array<CutQuadratureRule, 2> makeConservativeSplitCutVolumeQuadrature(
    Real parent_measure,
    Real negative_fraction,
    const std::array<Real, 3>& representative_point,
    const std::array<Real, 3>& interface_normal,
    const CutQuadratureConstructionPolicy& policy,
    CutQuadratureProvenance provenance) {
    negative_fraction = std::max(Real{0.0}, std::min(Real{1.0}, negative_fraction));
    std::array<CutQuadratureRule, 2> rules{};
    for (int i = 0; i < 2; ++i) {
        auto& rule = rules[static_cast<std::size_t>(i)];
        rule.kind = CutQuadratureKind::Volume;
        rule.side = i == 0 ? CutIntegrationSide::Negative : CutIntegrationSide::Positive;
        rule.parent_measure = std::max(Real{0.0}, parent_measure);
        rule.volume_fraction = i == 0 ? negative_fraction : Real{1.0} - negative_fraction;
        rule.measure = rule.parent_measure * rule.volume_fraction;
        rule.exact_for_constants = true;
        rule.exact_polynomial_order = std::max(0, policy.polynomial_order);
        rule.policy = policy;
        rule.policy.kind = CutQuadratureConstructionKind::ConservativeMomentFit;
        rule.provenance = provenance;
        rule.provenance.construction = CutQuadratureConstructionKind::ConservativeMomentFit;
        rule.provenance_id = provenance.embedded_geometry_id;
        rule.frame = provenance.frame;
        if (rule.measure > Real{0.0}) {
            CutQuadraturePoint qp;
            qp.weight = rule.measure;
            qp.point = representative_point;
            qp.normal = rule.side == CutIntegrationSide::Negative
                            ? unit_or_default(interface_normal)
                            : scale(unit_or_default(interface_normal), Real{-1.0});
            rule.points.push_back(qp);
        }
    }
    return rules;
}

CutMeasureConservationDiagnostic checkCutVolumeConservation(
    const CutQuadratureRule& negative,
    const CutQuadratureRule& positive,
    Real tolerance) {
    CutMeasureConservationDiagnostic diagnostic;
    diagnostic.parent_measure = std::max(negative.parent_measure, positive.parent_measure);
    diagnostic.negative_measure = negative.measure;
    diagnostic.positive_measure = positive.measure;
    diagnostic.residual =
        std::abs((negative.measure + positive.measure) - diagnostic.parent_measure);
    diagnostic.tolerance = tolerance;
    diagnostic.ok = diagnostic.residual <= tolerance;
    return diagnostic;
}

CutGeometrySensitivity makeAxisAlignedBoxCutSensitivity(
    const std::array<Real, 3>& min_corner,
    const std::array<Real, 3>& max_corner,
    int axis,
    Real cut_coordinate,
    CutIntegrationSide side) {
    CutGeometrySensitivity sensitivity;
    if (side == CutIntegrationSide::Interface) {
        sensitivity.capability_diagnostic = "volume-fraction sensitivity requires a side";
        return sensitivity;
    }
    validate_axis(axis);
    const auto ax = static_cast<std::size_t>(axis);
    const Real parent = box_measure(min_corner, max_corner);
    if (parent <= Real{0.0} || cut_coordinate < min_corner[ax] || cut_coordinate > max_corner[ax]) {
        sensitivity.capability_diagnostic = "cut coordinate is outside a nondegenerate box";
        return sensitivity;
    }
    Real cross_measure = Real{1.0};
    for (int d = 0; d < 3; ++d) {
        if (d != axis) {
            cross_measure *= std::max(Real{0.0}, max_corner[static_cast<std::size_t>(d)] -
                                                   min_corner[static_cast<std::size_t>(d)]);
        }
    }
    const Real sign = side == CutIntegrationSide::Negative ? Real{1.0} : Real{-1.0};
    sensitivity.available = true;
    sensitivity.d_location_d_plane_origin_diagonal[ax] = Real{1.0};
    sensitivity.d_location_d_mesh_point_diagonal = {{Real{1.0}, Real{1.0}, Real{1.0}}};
    sensitivity.d_measure_d_plane_origin[ax] = sign * cross_measure;
    sensitivity.d_volume_fraction_d_cut_coordinate = sign * cross_measure / parent;
    sensitivity.d_normal_d_plane_normal_diagonal = {{Real{1.0}, Real{1.0}, Real{1.0}}};
    return sensitivity;
}

CutGeometrySensitivity makePlaneCutLocationSensitivity(
    const std::array<Real, 3>& plane_normal) {
    CutGeometrySensitivity sensitivity;
    const Real len = norm(plane_normal);
    if (len <= Real{1.0e-30}) {
        sensitivity.capability_diagnostic = "plane normal is degenerate";
        return sensitivity;
    }
    const auto n = scale(plane_normal, Real{1.0} / len);
    sensitivity.available = true;
    sensitivity.d_location_d_plane_origin_diagonal = {{n[0] * n[0], n[1] * n[1], n[2] * n[2]}};
    sensitivity.d_location_d_mesh_point_diagonal = {{
        Real{1.0} - n[0] * n[0],
        Real{1.0} - n[1] * n[1],
        Real{1.0} - n[2] * n[2]}};
    sensitivity.d_normal_d_plane_normal_diagonal = {{Real{1.0}, Real{1.0}, Real{1.0}}};
    return sensitivity;
}

CutQuadratureDiagnostic diagnoseCutQuadrature(
    const CutQuadratureRule& rule,
    const CutQuadratureValidityPolicy& policy) {
    CutQuadratureDiagnostic diagnostic;
    if (!std::isfinite(rule.measure) ||
        !std::isfinite(rule.parent_measure) ||
        !std::isfinite(rule.volume_fraction) ||
        rule.measure < Real{0.0} ||
        rule.parent_measure < Real{0.0}) {
        diagnostic.ok = false;
        diagnostic.degenerate = true;
        diagnostic.nonfinite_geometry = true;
        diagnostic.messages.push_back("cut quadrature rule has invalid measure metadata");
        return diagnostic;
    }

    Real weight_sum = Real{0.0};
    bool have_reference_normal = false;
    std::array<Real, 3> reference_normal{{0.0, 0.0, 0.0}};
    for (const auto& qp : rule.points) {
        const bool finite_point = std::isfinite(qp.point[0]) &&
                                  std::isfinite(qp.point[1]) &&
                                  std::isfinite(qp.point[2]) &&
                                  std::isfinite(qp.normal[0]) &&
                                  std::isfinite(qp.normal[1]) &&
                                  std::isfinite(qp.normal[2]) &&
                                  std::isfinite(qp.weight);
        if (!finite_point) {
            diagnostic.ok = false;
            diagnostic.degenerate = true;
            diagnostic.nonfinite_geometry = true;
            diagnostic.messages.push_back("cut quadrature contains a non-finite point or weight");
            break;
        }
        if (qp.weight <= Real{0.0}) {
            diagnostic.degenerate = true;
            diagnostic.messages.push_back("cut quadrature contains a non-positive point weight");
        }
        weight_sum += qp.weight;
        const Real normal_length = norm(qp.normal);
        if (rule.curved_geometry && normal_length <= Real{1.0e-30}) {
            diagnostic.degenerate = true;
            diagnostic.inconsistent_normals = true;
            diagnostic.messages.push_back("curved cut quadrature contains a degenerate normal");
        }
        if (rule.curved_geometry && normal_length > Real{1.0e-30}) {
            const auto n = scale(qp.normal, Real{1.0} / normal_length);
            if (!have_reference_normal) {
                reference_normal = n;
                have_reference_normal = true;
            } else if (dot(reference_normal, n) < Real{-0.25}) {
                diagnostic.inconsistent_normals = true;
                diagnostic.degenerate = true;
                diagnostic.messages.push_back("curved cut quadrature contains inverted normals");
            }
        }
    }
    if (rule.curved_geometry &&
        rule.kind == CutQuadratureKind::Interface &&
        rule.points.empty()) {
        diagnostic.degenerate = true;
        diagnostic.messages.push_back("curved interface quadrature has no positive-weight points");
    }
    if (rule.measure < policy.min_measure) {
        diagnostic.degenerate = true;
        diagnostic.messages.push_back("cut quadrature measure is below minimum");
    }
    if (rule.kind == CutQuadratureKind::Volume &&
        rule.volume_fraction < policy.min_fraction) {
        diagnostic.small_fraction = true;
        diagnostic.messages.push_back("cut volume fraction is below minimum");
    }
    if (policy.reject_degenerate && diagnostic.degenerate) {
        diagnostic.ok = false;
    }
    if (rule.parent_measure > Real{0.0} &&
        rule.kind == CutQuadratureKind::Volume &&
        std::abs(rule.measure - rule.volume_fraction * rule.parent_measure) >
            std::max(policy.min_measure, Real{1.0e-12} * rule.parent_measure)) {
        diagnostic.conservation_failure = true;
        diagnostic.ok = false;
        diagnostic.messages.push_back("cut quadrature measure and volume fraction are inconsistent");
    }
    if (!rule.points.empty()) {
        const Real scale_measure = std::max({Real{1.0}, std::abs(rule.measure), std::abs(weight_sum)});
        const Real tolerance = std::max(policy.min_measure, Real{1.0e-12} * scale_measure);
        if (std::abs(weight_sum - rule.measure) > tolerance) {
            diagnostic.conservation_failure = true;
            diagnostic.ok = false;
            diagnostic.messages.push_back("cut quadrature point weights do not sum to the rule measure");
        }
    }
    return diagnostic;
}

} // namespace geometry
} // namespace FE
} // namespace svmp
