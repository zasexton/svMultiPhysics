/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_GEOMETRY_CUTQUADRATURE_H
#define SVMP_FE_GEOMETRY_CUTQUADRATURE_H

/**
 * @file CutQuadrature.h
 * @brief Physics-neutral quadrature data for cut cells and embedded interfaces.
 */

#include "Core/Types.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace geometry {

enum class CutIntegrationSide : std::uint8_t {
    Negative,
    Positive,
    Interface
};

enum class CutQuadratureKind : std::uint8_t {
    Volume,
    Face,
    Interface
};

struct CutQuadraturePoint {
    std::array<Real, 3> point{{0.0, 0.0, 0.0}};
    std::array<Real, 3> normal{{0.0, 0.0, 0.0}};
    Real weight{0.0};
};

struct CutQuadratureRule {
    CutQuadratureKind kind{CutQuadratureKind::Volume};
    CutIntegrationSide side{CutIntegrationSide::Negative};
    std::vector<CutQuadraturePoint> points{};
    Real measure{0.0};
    Real parent_measure{0.0};
    Real volume_fraction{0.0};
    bool exact_for_constants{true};
    std::string provenance_id{};
};

struct CutQuadratureValidityPolicy {
    Real min_fraction{1.0e-10};
    Real min_measure{1.0e-14};
    bool reject_degenerate{true};
};

struct CutQuadratureDiagnostic {
    bool ok{true};
    bool small_fraction{false};
    bool degenerate{false};
    std::vector<std::string> messages{};
};

[[nodiscard]] CutQuadratureRule makeAxisAlignedBoxCutVolumeQuadrature(
    const std::array<Real, 3>& min_corner,
    const std::array<Real, 3>& max_corner,
    int axis,
    Real cut_coordinate,
    CutIntegrationSide side,
    std::string provenance_id = {});

[[nodiscard]] CutQuadratureRule makeAxisAlignedBoxCutInterfaceQuadrature(
    const std::array<Real, 3>& min_corner,
    const std::array<Real, 3>& max_corner,
    int axis,
    Real cut_coordinate,
    std::string provenance_id = {});

[[nodiscard]] CutQuadratureRule makeSegmentCutFaceQuadrature(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b,
    const std::array<Real, 3>& plane_origin,
    const std::array<Real, 3>& plane_normal,
    CutIntegrationSide side,
    std::string provenance_id = {});

[[nodiscard]] CutQuadratureDiagnostic diagnoseCutQuadrature(
    const CutQuadratureRule& rule,
    const CutQuadratureValidityPolicy& policy = {});

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_CUTQUADRATURE_H
