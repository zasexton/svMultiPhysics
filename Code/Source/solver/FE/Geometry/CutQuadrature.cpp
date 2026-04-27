/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Geometry/CutQuadrature.h"

#include <algorithm>
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
    rule.provenance_id = std::move(provenance_id);
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
    rule.provenance_id = std::move(provenance_id);
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
    Real t = Real{0.0};
    if (std::abs(da - db) > Real{1.0e-30}) {
        t = da / (da - db);
    }
    t = std::max(Real{0.0}, std::min(Real{1.0}, t));
    const auto hit = add(scale(a, Real{1.0} - t), scale(b, t));

    std::array<Real, 3> seg_min = a;
    std::array<Real, 3> seg_max = b;
    if (side == CutIntegrationSide::Negative) {
        if (da <= Real{0.0}) {
            seg_max = hit;
        } else {
            seg_min = hit;
        }
    } else {
        if (da >= Real{0.0}) {
            seg_max = hit;
        } else {
            seg_min = hit;
        }
    }

    const Real parent = norm(sub(b, a));
    const Real measure = norm(sub(seg_max, seg_min));
    CutQuadratureRule rule;
    rule.kind = CutQuadratureKind::Face;
    rule.side = side;
    rule.measure = measure;
    rule.parent_measure = parent;
    rule.volume_fraction = parent > Real{0.0} ? measure / parent : Real{0.0};
    rule.provenance_id = std::move(provenance_id);
    if (measure > Real{0.0}) {
        CutQuadraturePoint qp;
        qp.weight = measure;
        qp.point = scale(add(seg_min, seg_max), Real{0.5});
        qp.normal = side == CutIntegrationSide::Negative ? n : scale(n, Real{-1.0});
        rule.points.push_back(qp);
    }
    return rule;
}

CutQuadratureDiagnostic diagnoseCutQuadrature(
    const CutQuadratureRule& rule,
    const CutQuadratureValidityPolicy& policy) {
    CutQuadratureDiagnostic diagnostic;
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
    return diagnostic;
}

} // namespace geometry
} // namespace FE
} // namespace svmp
