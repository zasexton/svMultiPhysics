/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_QUADRATURE_IMPLICITBOUNDARYINTERSECTIONQUADRATURE_H
#define SVMP_FE_QUADRATURE_IMPLICITBOUNDARYINTERSECTIONQUADRATURE_H

/**
 * @file ImplicitBoundaryIntersectionQuadrature.h
 * @brief Physics-neutral quadrature for implicit-field intersections on
 *        boundary subentities.
 */

#include "Core/Types.h"

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace quadrature {

enum class ImplicitBoundaryIntersectionKind : std::uint8_t {
    Point,
    Segment
};

enum class ImplicitBoundaryIntersectionStatus : std::uint8_t {
    Empty,
    Active,
    VertexTouch,
    EdgeAlignedZero,
    FullyDegenerateSubentity,
    VanishingMeasure,
    Ambiguous,
    UnsupportedElement,
    InvalidInput
};

struct ImplicitBoundaryIntersectionTolerance {
    Real zero{1.0e-12};
    Real duplicate{1.0e-12};
    Real measure{1.0e-14};
};

struct ImplicitBoundaryIntersectionRequest {
    ElementType parent_element{ElementType::Unknown};
    MeshIndex parent_cell{static_cast<MeshIndex>(-1)};
    int parent_dimension{0};
    LocalIndex local_subentity{INVALID_LOCAL_INDEX};
    std::vector<std::array<Real, 3>> parent_node_coordinates{};
    std::vector<Real> scalar_values{};
    Real isovalue{0.0};
    int quadrature_order{1};
    ImplicitBoundaryIntersectionTolerance tolerance{};

    // Optional evaluation hooks for linearly reconstructed parent cells. The
    // intersection geometry and returned differential data are always in the
    // canonical parent-reference frame. High-order parent elements fail
    // closed until an isoparametric intersection implementation is available.
    std::function<Real(const std::array<Real, 3>&)> scalar_evaluator{};
    std::function<std::array<Real, 3>(const std::array<Real, 3>&)>
        physical_mapping{};
};

struct ImplicitBoundaryIntersectionQuadraturePoint {
    std::array<Real, 3> parent_reference_coordinate{{0.0, 0.0, 0.0}};
    // Auxiliary diagnostic coordinate only; it does not define the frame of
    // the weight or directions below.
    std::array<Real, 3> physical_coordinate{{0.0, 0.0, 0.0}};
    // Reference-frame directions and codimension-two weight. The parent FE
    // assembler maps these exactly once.
    std::array<Real, 3> implicit_normal{{1.0, 0.0, 0.0}};
    std::array<Real, 3> boundary_normal{{0.0, 1.0, 0.0}};
    std::array<Real, 3> tangent{{1.0, 0.0, 0.0}};
    Real weight{0.0};
    Real scalar_residual{0.0};
    Real gradient_norm{0.0};
};

struct ImplicitBoundaryIntersectionFragment {
    ImplicitBoundaryIntersectionKind kind{
        ImplicitBoundaryIntersectionKind::Point};
    ImplicitBoundaryIntersectionStatus status{
        ImplicitBoundaryIntersectionStatus::Empty};
    MeshIndex parent_cell{static_cast<MeshIndex>(-1)};
    LocalIndex local_subentity{INVALID_LOCAL_INDEX};
    // Reference d-2 measure (one for an active 2D point, reference length for
    // a 3D segment).
    Real measure{0.0};
    std::array<Real, 3> implicit_normal{{1.0, 0.0, 0.0}};
    std::array<Real, 3> boundary_normal{{0.0, 1.0, 0.0}};
    std::array<Real, 3> tangent{{1.0, 0.0, 0.0}};
    std::string diagnostic{};
    std::vector<ImplicitBoundaryIntersectionQuadraturePoint>
        quadrature_points{};

    [[nodiscard]] bool active() const noexcept;
};

struct ImplicitBoundaryIntersectionResult {
    ElementType parent_element{ElementType::Unknown};
    MeshIndex parent_cell{static_cast<MeshIndex>(-1)};
    LocalIndex local_subentity{INVALID_LOCAL_INDEX};
    ImplicitBoundaryIntersectionStatus status{
        ImplicitBoundaryIntersectionStatus::Empty};
    std::vector<ImplicitBoundaryIntersectionFragment> fragments{};

    [[nodiscard]] bool hasActiveFragments() const noexcept;
    [[nodiscard]] std::size_t quadraturePointCount() const noexcept;
    [[nodiscard]] Real measure() const noexcept;
};

[[nodiscard]] const char* implicitBoundaryIntersectionStatusName(
    ImplicitBoundaryIntersectionStatus status) noexcept;

[[nodiscard]] bool supportsImplicitBoundaryIntersectionQuadrature(
    ElementType parent_element) noexcept;

[[nodiscard]] std::vector<LocalIndex> implicitBoundarySubentityCornerIndices(
    ElementType parent_element,
    LocalIndex local_subentity);

[[nodiscard]] ImplicitBoundaryIntersectionResult
buildImplicitBoundaryIntersectionQuadrature(
    const ImplicitBoundaryIntersectionRequest& request);

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_IMPLICITBOUNDARYINTERSECTIONQUADRATURE_H
