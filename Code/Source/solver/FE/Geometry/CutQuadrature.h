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
namespace interfaces {
enum class LevelSetCoefficientClassificationPolicy : std::uint8_t;
}
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

enum class CutQuadratureConstructionKind : std::uint8_t {
    AxisAlignedBoxClip,
    SegmentClip,
    PolygonInterface,
    ConservativeMomentFit,
    TopologySubdivision,
    CurvedInterface,
    MomentFittedImplicit,
    CurvedTopologySubdivision
};

enum class CutGeometryFrame : std::uint8_t {
    Reference,
    Current
};

struct CutQuadraturePoint {
    std::array<Real, 3> point{{0.0, 0.0, 0.0}};
    std::array<Real, 3> normal{{0.0, 0.0, 0.0}};
    std::array<Real, 3> boundary_normal{{0.0, 0.0, 0.0}};
    std::array<Real, 3> tangent{{0.0, 0.0, 0.0}};
    Real weight{0.0};
    std::array<Real, 3> parent_coordinate{{0.0, 0.0, 0.0}};
    Real reference_measure_factor{0.0};
    Real level_set_residual{0.0};
    Real gradient_norm{0.0};
};

struct CutQuadratureProvenance {
    std::string embedded_geometry_id{};
    std::string cut_topology_id{};
    // Rank-local parent id used to address assembly storage.
    MeshIndex parent_entity{static_cast<MeshIndex>(-1)};
    MeshIndex parent_boundary_entity{static_cast<MeshIndex>(-1)};
    // Partition-independent identities and the authoritative owner.  These
    // remain distinct from parent_entity because assembly requires local ids.
    GlobalIndex parent_entity_global_id{INVALID_GLOBAL_INDEX};
    GlobalIndex parent_boundary_entity_global_id{INVALID_GLOBAL_INDEX};
    int owner_rank{-1};
    int marker{-1};
    std::uint64_t cut_topology_revision{0};
    std::uint64_t predicate_policy_key{0};
    interfaces::LevelSetCoefficientClassificationPolicy
        coefficient_classification_policy{};
    Real coefficient_classification_band{0.0};
    std::uint64_t source_value_revision{0};
    // Immediate generated-geometry parent, when this rule is derived from
    // another retained fragment (for example a contact rule from a surface).
    std::uint64_t source_stable_id{0};
    CutQuadratureConstructionKind construction{CutQuadratureConstructionKind::AxisAlignedBoxClip};
    CutGeometryFrame frame{CutGeometryFrame::Reference};
    std::string implicit_geometry_mode{};
    std::string implicit_quadrature_backend{};
    std::string selected_implicit_quadrature_backend{};
    std::string implicit_fallback_policy{};
    std::string implicit_fallback_status{};
    std::string geometry_tangent_policy{};
    Real implicit_cut_root_tolerance{0.0};
    Real implicit_cut_root_coordinate_tolerance{0.0};
    int implicit_cut_root_max_iterations{0};
    int requested_quadrature_order{-1};
    int achieved_quadrature_order{-1};
    // Complete immutable free-surface snapshot binding.  Standalone cut
    // rules leave this zero; importing a validated snapshot stamps every
    // retained rule with the finalized, content-addressed revision key.
    std::uint64_t free_surface_snapshot_revision_key{0};
};

struct CutQuadratureConstructionPolicy {
    CutQuadratureConstructionKind kind{CutQuadratureConstructionKind::AxisAlignedBoxClip};
    int polynomial_order{0};
    bool moment_fitted{false};
    Real tolerance{1.0e-12};
    std::string name{"constant-exact"};
};

struct CutQuadratureRule {
    CutQuadratureKind kind{CutQuadratureKind::Volume};
    CutIntegrationSide side{CutIntegrationSide::Negative};
    /**
     * Dimension of the integration set in the parent cell. A negative value
     * selects the legacy inference (cell dimension for Volume and parent
     * dimension minus one for Face/Interface). Generated codimension-two
     * rules must set this explicitly: zero for a point in 2D and one for a
     * line in 3D.
     */
    int geometric_dimension{-1};
    std::vector<CutQuadraturePoint> points{};
    Real measure{0.0};
    Real parent_measure{0.0};
    Real volume_fraction{0.0};
    bool exact_for_constants{true};
    int exact_polynomial_order{0};
    CutQuadratureConstructionPolicy policy{};
    CutQuadratureProvenance provenance{};
    std::string provenance_id{};
    CutGeometryFrame frame{CutGeometryFrame::Reference};
    bool curved_geometry{false};
    bool full_cell_equivalent{false};
};

using CutGeometryJacobian = std::array<std::array<Real, 3>, 3>;

/** Physical geometry obtained by mapping one reference codimension-two point. */
struct MappedCutCodimensionTwoGeometry {
    std::array<Real, 3> interface_normal{{1.0, 0.0, 0.0}};
    std::array<Real, 3> boundary_normal{{0.0, 1.0, 0.0}};
    std::array<Real, 3> tangent{{1.0, 0.0, 0.0}};
    Real measure_scale{1.0};
    Real weight{0.0};
};

/**
 * Map a strict reference-frame d-2 quadrature point to physical geometry.
 * In 2D the point weight is invariant. In 3D the line weight receives
 * `|J t_ref|`; covectors (both normals) receive `J^{-T}` and are normalized.
 */
[[nodiscard]] MappedCutCodimensionTwoGeometry
mapReferenceCutCodimensionTwoGeometry(
    const CutQuadraturePoint& point,
    int parent_dimension,
    const CutGeometryJacobian& jacobian,
    const CutGeometryJacobian& inverse_jacobian);

struct CutQuadratureValidityPolicy {
    Real min_fraction{1.0e-10};
    Real min_measure{1.0e-14};
    bool reject_degenerate{true};
};

struct CutQuadratureDiagnostic {
    bool ok{true};
    bool small_fraction{false};
    bool degenerate{false};
    bool conservation_failure{false};
    bool nonfinite_geometry{false};
    bool inconsistent_normals{false};
    std::vector<std::string> messages{};
};

struct CutMeasureConservationDiagnostic {
    bool ok{true};
    Real parent_measure{0.0};
    Real negative_measure{0.0};
    Real positive_measure{0.0};
    Real residual{0.0};
    Real tolerance{1.0e-12};
};

struct CutGeometrySensitivity {
    bool available{false};
    std::array<Real, 3> d_location_d_plane_origin_diagonal{{0.0, 0.0, 0.0}};
    std::array<Real, 3> d_location_d_mesh_point_diagonal{{0.0, 0.0, 0.0}};
    std::array<Real, 3> d_measure_d_plane_origin{{0.0, 0.0, 0.0}};
    std::array<Real, 3> d_measure_d_plane_normal{{0.0, 0.0, 0.0}};
    Real d_volume_fraction_d_cut_coordinate{0.0};
    std::array<Real, 3> d_normal_d_plane_normal_diagonal{{0.0, 0.0, 0.0}};
    std::string capability_diagnostic{};
};

struct CutTopologyQuadratureInput {
    Real parent_measure{0.0};
    Real negative_fraction{0.0};
    std::array<Real, 3> representative_point{{0.0, 0.0, 0.0}};
    std::array<Real, 3> interface_normal{{1.0, 0.0, 0.0}};
    std::vector<std::array<Real, 3>> ordered_interface_points{};
    CutQuadratureConstructionPolicy policy{};
    CutQuadratureProvenance provenance{};
    CutGeometryFrame frame{CutGeometryFrame::Reference};
    bool curved_geometry{false};
};

struct CutTopologySubcellInput {
    std::string topology_id{};
    std::vector<std::array<Real, 3>> points{};
    Real measure{0.0};
    std::array<Real, 3> centroid{{0.0, 0.0, 0.0}};
    bool curved_geometry{false};
    bool measure_from_isoparametric_quadrature{false};
};

struct CutClosedTopologyQuadratureInput {
    Real parent_measure{0.0};
    std::vector<CutTopologySubcellInput> negative_subcells{};
    std::vector<CutTopologySubcellInput> positive_subcells{};
    std::vector<std::array<Real, 3>> ordered_interface_points{};
    std::vector<CutQuadraturePoint> curved_interface_points{};
    std::array<Real, 3> interface_normal{{1.0, 0.0, 0.0}};
    CutQuadratureConstructionPolicy policy{};
    CutQuadratureProvenance provenance{};
    CutGeometryFrame frame{CutGeometryFrame::Reference};
    bool curved_geometry{false};
};

struct CutTopologyQuadratureRules {
    CutQuadratureRule negative_volume{};
    CutQuadratureRule positive_volume{};
    CutQuadratureRule interface_rule{};
    CutMeasureConservationDiagnostic conservation{};
    CutQuadratureDiagnostic diagnostic{};
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

[[nodiscard]] CutQuadratureRule makePolygonInterfaceQuadrature(
    const std::vector<std::array<Real, 3>>& ordered_points,
    const std::array<Real, 3>& normal,
    const CutQuadratureConstructionPolicy& policy = {},
    CutQuadratureProvenance provenance = {});

[[nodiscard]] CutTopologyQuadratureRules makeTopologyDerivedCutQuadrature(
    const CutTopologyQuadratureInput& input);

[[nodiscard]] CutTopologyQuadratureRules makeClosedTopologyCutQuadrature(
    const CutClosedTopologyQuadratureInput& input);

[[nodiscard]] CutQuadratureRule makeCurvedInterfaceQuadrature(
    const std::vector<CutQuadraturePoint>& points,
    CutGeometryFrame frame,
    const CutQuadratureConstructionPolicy& policy = {},
    CutQuadratureProvenance provenance = {});

[[nodiscard]] CutQuadratureRule makeMomentFittedCutVolumeQuadrature(
    Real parent_measure,
    Real side_fraction,
    CutIntegrationSide side,
    const std::array<Real, 3>& moment_point,
    const std::array<Real, 3>& interface_normal,
    int exact_polynomial_order,
    CutQuadratureProvenance provenance = {});

[[nodiscard]] std::array<CutQuadratureRule, 2> makeConservativeSplitCutVolumeQuadrature(
    Real parent_measure,
    Real negative_fraction,
    const std::array<Real, 3>& representative_point,
    const std::array<Real, 3>& interface_normal,
    const CutQuadratureConstructionPolicy& policy = {},
    CutQuadratureProvenance provenance = {});

[[nodiscard]] CutMeasureConservationDiagnostic checkCutVolumeConservation(
    const CutQuadratureRule& negative,
    const CutQuadratureRule& positive,
    Real tolerance = 1.0e-12);

[[nodiscard]] CutGeometrySensitivity makeAxisAlignedBoxCutSensitivity(
    const std::array<Real, 3>& min_corner,
    const std::array<Real, 3>& max_corner,
    int axis,
    Real cut_coordinate,
    CutIntegrationSide side);

[[nodiscard]] CutGeometrySensitivity makePlaneCutLocationSensitivity(
    const std::array<Real, 3>& plane_normal);

[[nodiscard]] CutQuadratureDiagnostic diagnoseCutQuadrature(
    const CutQuadratureRule& rule,
    const CutQuadratureValidityPolicy& policy = {});

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_CUTQUADRATURE_H
