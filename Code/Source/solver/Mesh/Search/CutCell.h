/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_SEARCH_CUT_CELL_H
#define SVMP_SEARCH_CUT_CELL_H

/**
 * @file CutCell.h
 * @brief Physics-agnostic embedded-geometry and cut-entity classification.
 *
 * This module owns geometric classification and provenance for unfitted
 * interfaces. It deliberately stops at geometry, kinematics, revision state,
 * and restart metadata; weak interface terms and physical jump conditions
 * belong in Physics modules.
 */

#include "../Core/MeshTypes.h"
#include "../Validation/MovingGeometryValidity.h"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp {

class MeshBase;
class DistributedMesh;

namespace search {

enum class EmbeddedGeometryKind : std::uint8_t {
  Plane,
  Sphere,
  SignedDistanceCallback,
  LevelSetField,
  TriangulatedSurface,
  BooleanComposite
};

enum class EmbeddedGeometryBooleanOperation : std::uint8_t {
  Union,
  Intersection,
  Difference
};

enum class CutEntityKind : std::uint8_t {
  Cell,
  Face,
  Edge
};

enum class CutClassification : std::uint8_t {
  Negative,
  Positive,
  Cut,
  Degenerate
};

enum class CutClassificationState : std::uint8_t {
  Empty,
  Trial,
  Committed,
  RolledBack
};

enum class EmbeddedKinematicConstraintKind : std::uint8_t {
  PrescribedMotion,
  ManifoldFollowing,
  RelationMap
};

enum class CutTopologyEntityKind : std::uint8_t {
  CutVertex,
  CutEdge,
  InterfacePolygon,
  SideRegion
};

enum class CutTopologySide : std::uint8_t {
  Negative,
  Positive,
  Interface
};

enum class CutSupportStatus : std::uint8_t {
  Supported,
  ImplementedUnqualified,
  Unsupported
};

enum class CutCurvedArrangementMode : std::uint8_t {
  LinearizedSurrogate,
  TrueArrangement
};

struct EmbeddedRegionProvenance {
  std::string persistent_id{};
  std::string name{};
  label_t physical_label{INVALID_LABEL};
  std::uint64_t provenance_epoch{0};

  [[nodiscard]] bool empty() const noexcept { return persistent_id.empty() && name.empty(); }
};

struct EmbeddedGeometryRevisionState {
  std::uint64_t geometry_epoch{0};
  std::uint64_t field_layout_revision{0};
  std::uint64_t field_value_revision{0};
  std::uint64_t source_surface_revision{0};
  std::uint64_t provenance_revision{0};
  std::uint64_t kinematic_constraint_revision{0};

  [[nodiscard]] std::uint64_t revision_key() const noexcept;
};

struct EmbeddedRevisionSnapshot {
  Configuration configuration{Configuration::Reference};
  std::uint64_t geometry_revision{0};
  std::uint64_t topology_revision{0};
  std::uint64_t ownership_revision{0};
  std::uint64_t numbering_revision{0};
  std::uint64_t label_revision{0};
  std::uint64_t active_configuration_epoch{0};
  std::uint64_t embedded_geometry_epoch{0};
  std::uint64_t embedded_field_layout_revision{0};
  std::uint64_t embedded_field_value_revision{0};
  std::uint64_t embedded_source_surface_revision{0};
  std::uint64_t embedded_provenance_revision{0};
  std::uint64_t embedded_constraint_epoch{0};
  std::uint64_t fe_layout_revision{0};

  [[nodiscard]] static EmbeddedRevisionSnapshot capture(
      const MeshBase& mesh,
      Configuration configuration,
      std::uint64_t embedded_geometry_epoch,
      std::uint64_t embedded_constraint_epoch = 0,
      std::uint64_t fe_layout_revision = 0);

  [[nodiscard]] static EmbeddedRevisionSnapshot capture(
      const MeshBase& mesh,
      Configuration configuration,
      const EmbeddedGeometryRevisionState& embedded_revisions,
      std::uint64_t fe_layout_revision = 0);

  [[nodiscard]] bool matches(const MeshBase& mesh,
                             Configuration configuration,
                             std::uint64_t embedded_geometry_epoch,
                             std::uint64_t embedded_constraint_epoch = 0,
                             std::uint64_t fe_layout_revision = 0) const noexcept;
  [[nodiscard]] bool matches(const MeshBase& mesh,
                             Configuration configuration,
                             const EmbeddedGeometryRevisionState& embedded_revisions,
                             std::uint64_t fe_layout_revision = 0) const noexcept;
  [[nodiscard]] std::uint64_t revision_key() const noexcept;
};

struct EmbeddedGeometryQueryDiagnostic {
  bool ok{true};
  std::vector<std::string> messages{};
};

struct CutOperationDiagnostic {
  bool ok{true};
  std::string operation{};
  std::uint64_t predicate_policy_key{0};
  EmbeddedRevisionSnapshot mesh_and_embedded_revision{};
  EmbeddedGeometryRevisionState embedded_revision{};
  std::uint64_t fe_layout_revision{0};
  std::vector<std::string> messages{};
};

struct EmbeddedLevelSetSample {
  std::array<real_t, 3> point{{0.0, 0.0, 0.0}};
  real_t value{0.0};
  std::array<real_t, 3> gradient{{0.0, 0.0, 0.0}};
};

struct EmbeddedSurfaceTriangle {
  std::array<std::array<real_t, 3>, 3> vertices{};
  EmbeddedRegionProvenance provenance{};
};

struct EmbeddedGeometryDescriptor {
  EmbeddedGeometryKind kind{EmbeddedGeometryKind::Plane};
  Configuration configuration{Configuration::Reference};
  std::array<real_t, 3> origin{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> normal{{1.0, 0.0, 0.0}};
  real_t radius{1.0};
  std::uint64_t geometry_epoch{0};
  EmbeddedGeometryRevisionState revisions{};
  EmbeddedRegionProvenance provenance{};
  bool active{true};
  std::function<real_t(const std::array<real_t, 3>&)> signed_distance_callback{};
  std::function<std::array<real_t, 3>(const std::array<real_t, 3>&)> normal_callback{};
  std::function<std::array<real_t, 3>(const std::array<real_t, 3>&)> closest_point_callback{};
  std::vector<EmbeddedLevelSetSample> level_set_samples{};
  std::vector<EmbeddedSurfaceTriangle> surface_triangles{};
  EmbeddedGeometryBooleanOperation boolean_operation{EmbeddedGeometryBooleanOperation::Union};
  std::vector<EmbeddedGeometryDescriptor> children{};

  [[nodiscard]] real_t signed_distance(const std::array<real_t, 3>& point) const noexcept;
  [[nodiscard]] std::array<real_t, 3> outward_normal(const std::array<real_t, 3>& point) const noexcept;
  [[nodiscard]] std::array<real_t, 3> closest_point(const std::array<real_t, 3>& point) const noexcept;
  [[nodiscard]] EmbeddedGeometryRevisionState effective_revisions() const noexcept;
  [[nodiscard]] EmbeddedGeometryQueryDiagnostic diagnose_query_support() const;
};

struct EmbeddedGeometryRestartRecord {
  std::string persistent_id{};
  std::string name{};
  EmbeddedGeometryKind kind{EmbeddedGeometryKind::Plane};
  Configuration configuration{Configuration::Reference};
  std::array<real_t, 3> origin{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> normal{{1.0, 0.0, 0.0}};
  real_t radius{1.0};
  EmbeddedGeometryRevisionState revisions{};
  EmbeddedRegionProvenance provenance{};
  bool active{true};
  EmbeddedGeometryBooleanOperation boolean_operation{EmbeddedGeometryBooleanOperation::Union};
  std::vector<EmbeddedLevelSetSample> level_set_samples{};
  std::vector<EmbeddedSurfaceTriangle> surface_triangles{};
  std::vector<EmbeddedGeometryRestartRecord> children{};
  bool requires_application_reregistration{false};
};

struct EmbeddedCompositionChildRestartRecord {
  std::size_t depth{0};
  std::size_t child_ordinal{0};
  std::string parent_persistent_id{};
  EmbeddedRegionProvenance provenance{};
  EmbeddedGeometryKind kind{EmbeddedGeometryKind::Plane};
  EmbeddedGeometryBooleanOperation boolean_operation{EmbeddedGeometryBooleanOperation::Union};
  EmbeddedGeometryRevisionState revisions{};
};

struct EmbeddedKinematicConstraint {
  EmbeddedKinematicConstraintKind kind{EmbeddedKinematicConstraintKind::PrescribedMotion};
  std::string id{};
  std::string source_geometry_id{};
  std::string relation_map_id{};
  std::uint64_t constraint_epoch{0};
  EmbeddedRevisionSnapshot source_revision{};
  EmbeddedRegionProvenance provenance{};
  bool active{true};
};

struct CutIntersectionPoint {
  std::array<real_t, 3> point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> normal{{0.0, 0.0, 0.0}};
  real_t edge_fraction{0.0};
  index_t endpoint_a{INVALID_INDEX};
  index_t endpoint_b{INVALID_INDEX};
};

struct CutPredicatePolicy {
  validation::RobustPredicatePolicy robust{};
  std::string name{"default"};

  [[nodiscard]] real_t classification_tolerance() const noexcept {
    return robust.intersection_tolerance;
  }
  [[nodiscard]] real_t degenerate_tolerance() const noexcept {
    return robust.degenerate_tolerance;
  }
  [[nodiscard]] std::uint64_t revision_key() const noexcept;
};

struct CutEntityRecord {
  CutEntityKind kind{CutEntityKind::Cell};
  index_t entity{INVALID_INDEX};
  gid_t global_id{INVALID_GID};
  rank_t owner_rank{0};
  CutClassification classification{CutClassification::Degenerate};
  real_t min_signed_distance{0.0};
  real_t max_signed_distance{0.0};
  std::vector<CutIntersectionPoint> intersections{};
  EmbeddedRegionProvenance provenance{};
  std::uint64_t cut_topology_id{0};
  std::vector<std::string> diagnostics{};
};

struct CutClassificationOptions {
  Configuration configuration{Configuration::Reference};
  real_t tolerance{1.0e-12};
  CutPredicatePolicy predicate_policy{};
  bool classify_cells{true};
  bool classify_faces{true};
  bool classify_edges{true};
  std::uint64_t fe_layout_revision{0};
  std::vector<EmbeddedKinematicConstraint> kinematic_constraints{};
};

struct CutClassificationMap {
  std::string name{};
  EmbeddedGeometryDescriptor embedded_geometry{};
  CutClassificationOptions options{};
  EmbeddedRevisionSnapshot revision{};
  CutClassificationState state{CutClassificationState::Empty};
  std::vector<CutEntityRecord> cells{};
  std::vector<CutEntityRecord> faces{};
  std::vector<CutEntityRecord> edges{};
  std::vector<EmbeddedKinematicConstraint> kinematic_constraints{};
  std::vector<EmbeddedGeometryQueryDiagnostic> diagnostics{};

  [[nodiscard]] bool valid_for(const MeshBase& mesh) const noexcept;
  [[nodiscard]] std::uint64_t revision_key() const noexcept;
  void accept_trial() noexcept;
  void rollback_trial();
};

struct CutSideRegionRestartRecord {
  std::uint64_t stable_id{0};
  CutTopologySide side{CutTopologySide::Negative};
  index_t parent_cell{INVALID_INDEX};
  gid_t parent_cell_gid{INVALID_GID};
  real_t measure_estimate{0.0};
  real_t volume_fraction_estimate{0.0};
  EmbeddedRegionProvenance provenance{};
};

struct CutClassificationRestartRecord {
  std::string name{};
  EmbeddedRegionProvenance provenance{};
  EmbeddedGeometryKind embedded_kind{EmbeddedGeometryKind::Plane};
  bool is_composed_region{false};
  EmbeddedGeometryBooleanOperation composition_operation{EmbeddedGeometryBooleanOperation::Union};
  std::vector<EmbeddedCompositionChildRestartRecord> composition_children{};
  std::uint64_t revision_key{0};
  std::uint64_t embedded_geometry_epoch{0};
  std::uint64_t embedded_field_layout_revision{0};
  std::uint64_t embedded_field_value_revision{0};
  std::uint64_t embedded_source_surface_revision{0};
  std::uint64_t embedded_provenance_revision{0};
  std::uint64_t embedded_constraint_epoch{0};
  std::uint64_t fe_layout_revision{0};
  std::uint64_t predicate_policy_key{0};
  std::uint64_t cut_topology_revision{0};
  std::size_t cut_cell_count{0};
  std::size_t cut_face_count{0};
  std::size_t cut_edge_count{0};
  std::vector<CutSideRegionRestartRecord> side_regions{};
};

struct CutTopologyVertex {
  std::uint64_t stable_id{0};
  std::array<real_t, 3> point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> normal{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> parent_parametric_coordinate{{0.0, 0.0, 0.0}};
  index_t parent_cell{INVALID_INDEX};
  gid_t parent_cell_gid{INVALID_GID};
  index_t parent_edge{INVALID_INDEX};
  index_t endpoint_a{INVALID_INDEX};
  index_t endpoint_b{INVALID_INDEX};
  real_t edge_fraction{0.0};
  real_t parent_parametric_residual{0.0};
  bool has_parent_parametric_coordinate{false};
  bool parent_parametric_coordinate_valid{false};
  EmbeddedRegionProvenance provenance{};
};

struct CutTopologyEdge {
  std::uint64_t stable_id{0};
  std::uint64_t vertex_a{0};
  std::uint64_t vertex_b{0};
  index_t parent_cell{INVALID_INDEX};
  gid_t parent_cell_gid{INVALID_GID};
  EmbeddedRegionProvenance provenance{};
};

struct CutInterfacePolygon {
  std::uint64_t stable_id{0};
  index_t parent_cell{INVALID_INDEX};
  gid_t parent_cell_gid{INVALID_GID};
  std::vector<std::uint64_t> ordered_vertices{};
  std::array<real_t, 3> normal{{0.0, 0.0, 0.0}};
  EmbeddedRegionProvenance provenance{};
};

struct CutCurvedPatchRecord {
  std::uint64_t stable_id{0};
  index_t parent_cell{INVALID_INDEX};
  gid_t parent_cell_gid{INVALID_GID};
  CellFamily parent_family{CellFamily::Point};
  int geometry_order{1};
  EmbeddedGeometryKind embedded_kind{EmbeddedGeometryKind::Plane};
  Configuration configuration{Configuration::Reference};
  std::vector<std::uint64_t> ordered_vertices{};
  std::vector<std::array<real_t, 3>> parent_parametric_coordinates{};
  std::vector<std::array<real_t, 3>> physical_points{};
  std::vector<std::array<real_t, 3>> quadrature_points{};
  std::vector<std::array<real_t, 3>> quadrature_normals{};
  std::vector<real_t> quadrature_weights{};
  std::vector<std::array<real_t, 3>> quadrature_parent_parametric_coordinates{};
  std::vector<std::size_t> active_child_ordinals{};
  std::vector<EmbeddedRegionProvenance> active_child_provenance{};
  std::vector<std::size_t> predicate_fallback_child_ordinals{};
  real_t quadrature_measure{0.0};
  real_t max_parent_parametric_residual{0.0};
  bool parametric_coordinates_valid{false};
  bool exact_topology_available{false};
  bool linearized_surrogate{true};
  bool isoparametric_quadrature_available{false};
  bool predicate_fallback_used{false};
  bool predicate_fallback_tolerance_resolved{false};
  std::string construction_policy{"tessellated-curved-linearized-arrangement"};
  std::string predicate_fallback_policy{};
  std::string predicate_fallback_reason{};
  EmbeddedGeometryBooleanOperation composition_operation{EmbeddedGeometryBooleanOperation::Union};
  EmbeddedRegionProvenance provenance{};
};

struct CutQuadratureGeometrySensitivitySample {
  std::array<real_t, 3> parent_parametric_coordinate{{0.0, 0.0, 0.0}};
  std::vector<real_t> shape_values{};
  std::vector<std::array<real_t, 3>> shape_gradients{};
};

struct CutQuadratureGeometrySensitivityRecord {
  std::uint64_t stable_id{0};
  std::uint64_t source_stable_id{0};
  index_t parent_cell{INVALID_INDEX};
  gid_t parent_cell_gid{INVALID_GID};
  CellFamily parent_family{CellFamily::Point};
  int geometry_order{1};
  EmbeddedGeometryKind embedded_kind{EmbeddedGeometryKind::Plane};
  Configuration configuration{Configuration::Reference};
  std::string target_kind{"interface-quadrature"};
  std::string construction_policy{};
  std::vector<index_t> parent_geometry_dofs{};
  std::vector<CutQuadratureGeometrySensitivitySample> samples{};
  bool ad_compatible{false};
  bool location_sensitivity_available{false};
  bool jacobian_sensitivity_available{false};
  bool measure_sensitivity_available{false};
  bool normal_sensitivity_available{false};
  bool quadrature_weight_sensitivity_available{false};
  std::string capability_diagnostic{};
  EmbeddedRegionProvenance provenance{};
};

struct CutIntegrationVertex {
  std::uint64_t stable_id{0};
  std::array<real_t, 3> point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> parent_parametric_coordinate{{0.0, 0.0, 0.0}};
  index_t parent_geometry_dof{INVALID_INDEX};
  bool on_embedded_interface{false};
  bool synthetic{false};
  bool has_parent_parametric_coordinate{false};
  bool parent_parametric_coordinate_valid{false};
  bool curved_isoparametric{false};
  EmbeddedRegionProvenance provenance{};
};

struct CutIntegrationSubcell {
  std::uint64_t stable_id{0};
  CellFamily family{CellFamily::Point};
  std::vector<std::uint64_t> vertices{};
  std::vector<std::vector<std::uint64_t>> faces{};
  std::vector<std::array<real_t, 3>> parent_parametric_vertices{};
  real_t measure{0.0};
  real_t parent_parametric_measure{0.0};
  std::array<real_t, 3> centroid{{0.0, 0.0, 0.0}};
  bool closed_topology{false};
  bool curved_isoparametric{false};
  bool measure_from_isoparametric_quadrature{false};
  std::string construction_policy{"linear-topology-subdivision"};
  EmbeddedRegionProvenance provenance{};
};

struct CutSideRegion {
  std::uint64_t stable_id{0};
  CutTopologySide side{CutTopologySide::Negative};
  index_t parent_cell{INVALID_INDEX};
  gid_t parent_cell_gid{INVALID_GID};
  std::vector<index_t> parent_geometry_dofs{};
  std::vector<std::uint64_t> cut_vertices{};
  std::vector<std::uint64_t> integration_region_vertices{};
  std::vector<std::vector<std::uint64_t>> integration_region_faces{};
  std::vector<CutIntegrationVertex> integration_vertices{};
  std::vector<CutIntegrationSubcell> integration_subcells{};
  CellFamily integration_family{CellFamily::Point};
  real_t parent_measure{0.0};
  real_t measure_estimate{0.0};
  real_t volume_fraction_estimate{0.0};
  std::array<real_t, 3> centroid_estimate{{0.0, 0.0, 0.0}};
  bool measure_from_linear_topology{false};
  bool closed_integration_topology{false};
  bool curved_isoparametric_topology{false};
  EmbeddedRegionProvenance provenance{};
};

struct CutTopologyRecord {
  bool supported{true};
  bool linearized_cut_mode{true};
  std::uint64_t topology_revision{0};
  std::uint64_t predicate_policy_key{0};
  std::vector<CutTopologyVertex> vertices{};
  std::vector<CutTopologyEdge> edges{};
  std::vector<CutInterfacePolygon> interface_polygons{};
  std::vector<CutCurvedPatchRecord> curved_patches{};
  std::vector<CutSideRegion> side_regions{};
  std::vector<CutQuadratureGeometrySensitivityRecord> sensitivity_records{};
  std::vector<std::string> diagnostics{};
};

struct CutCurvedValidityDiagnostic {
  bool ok{true};
  bool has_nonfinite_geometry{false};
  bool has_folded_interface{false};
  bool has_degenerate_intersection{false};
  bool has_degenerate_polygon{false};
  bool has_inconsistent_side_region{false};
  bool has_open_subcell_topology{false};
  bool has_curved_sliver{false};
  bool has_invalid_curved_measure{false};
  bool requires_curved_geometry_support{false};
  std::vector<std::string> messages{};
};

struct CutCurvedValidityPolicy {
  real_t min_measure{1.0e-14};
  real_t min_fraction{1.0e-10};
  real_t closure_relative_tolerance{1.0e-8};
  real_t folding_tolerance{1.0e-12};
  bool reject_slivers{true};
};

struct CutTopologyOptions {
  Configuration configuration{Configuration::Reference};
  CutPredicatePolicy predicate_policy{};
  bool allow_linearized_high_order_geometry{true};
  CutCurvedArrangementMode curved_arrangement_mode{CutCurvedArrangementMode::LinearizedSurrogate};
  int true_curved_subdivision_refinement_level{-1};
  int true_curved_subdivision_max_refinement_level{-1};
  real_t true_curved_subdivision_curvature_threshold{0.0};
};

struct EmbeddedGeometryRegistrySnapshot {
  std::uint64_t registry_epoch{0};
  std::uint64_t geometry_revision{0};
  std::uint64_t field_layout_revision{0};
  std::uint64_t field_value_revision{0};
  std::uint64_t source_surface_revision{0};
  std::uint64_t provenance_revision{0};
  std::uint64_t kinematic_constraint_revision{0};
  std::size_t active_geometry_count{0};
};

struct CutDistributedEntityRecord {
  std::uint64_t stable_id{0};
  std::uint64_t cut_topology_id{0};
  CutTopologyEntityKind kind{CutTopologyEntityKind::CutVertex};
  CutTopologySide side{CutTopologySide::Interface};
  index_t parent_entity{INVALID_INDEX};
  gid_t parent_gid{INVALID_GID};
  rank_t owner_rank{0};
  std::string provenance_id{};
  std::array<real_t, 3> point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> normal{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> centroid{{0.0, 0.0, 0.0}};
  CellFamily integration_family{CellFamily::Point};
  real_t parent_measure{0.0};
  real_t measure{0.0};
  real_t volume_fraction{0.0};
  bool closed_topology{false};
  std::vector<std::uint64_t> vertex_ids{};
  std::vector<std::vector<std::uint64_t>> face_ids{};
  int geometry_order{1};
  EmbeddedGeometryKind embedded_kind{EmbeddedGeometryKind::Plane};
  Configuration configuration{Configuration::Reference};
  EmbeddedGeometryBooleanOperation composition_operation{EmbeddedGeometryBooleanOperation::Union};
  std::string construction_policy{};
  bool curved_isoparametric{false};
  bool parametric_coordinates_valid{false};
  bool exact_topology_available{false};
  bool linearized_surrogate{false};
  bool isoparametric_quadrature_available{false};
  std::vector<std::array<real_t, 3>> parent_parametric_coordinates{};
  std::vector<std::array<real_t, 3>> quadrature_parent_parametric_coordinates{};
  std::vector<real_t> quadrature_weights{};
  std::vector<std::size_t> active_child_ordinals{};
  std::vector<std::string> active_child_provenance_ids{};
  std::vector<std::size_t> predicate_fallback_child_ordinals{};
  std::size_t geometry_sensitivity_dof_count{0};
  std::size_t geometry_sensitivity_sample_count{0};
  bool geometry_sensitivity_ad_compatible{false};
  bool location_sensitivity_available{false};
  bool jacobian_sensitivity_available{false};
  bool measure_sensitivity_available{false};
  bool normal_sensitivity_available{false};
  bool quadrature_weight_sensitivity_available{false};
  bool predicate_fallback_used{false};
  bool predicate_fallback_tolerance_resolved{false};
  std::string predicate_fallback_policy{};
  std::string predicate_fallback_reason{};
};

struct CutDistributedExchangePacket {
  std::uint64_t revision_key{0};
  std::vector<CutDistributedEntityRecord> entities{};
  std::vector<std::string> diagnostics{};
};

struct CutDistributedRevisionSnapshot {
  Configuration configuration{Configuration::Reference};
  std::uint64_t geometry_revision{0};
  std::uint64_t topology_revision{0};
  std::uint64_t ownership_revision{0};
  std::uint64_t numbering_revision{0};
  std::uint64_t label_revision{0};
  std::uint64_t active_configuration_epoch{0};
  std::uint64_t classification_revision{0};
  std::uint64_t cut_topology_revision{0};
  std::uint64_t local_packet_revision{0};
  std::uint64_t exchanged_packet_revision{0};
  std::uint64_t distributed_revision{0};
  rank_t rank{0};
  rank_t world_size{1};

  [[nodiscard]] static CutDistributedRevisionSnapshot capture(
      const DistributedMesh& mesh,
      const CutClassificationMap& map,
      const CutTopologyRecord& topology,
      const CutDistributedExchangePacket& local_packet,
      const CutDistributedExchangePacket& exchanged_packet);

  [[nodiscard]] bool matches(const DistributedMesh& mesh,
                             const CutClassificationMap& map,
                             const CutTopologyRecord& topology) const noexcept;
  [[nodiscard]] std::uint64_t revision_key() const noexcept;
};

struct CutDistributedStateDiagnostic {
  bool ok{true};
  bool stale_revision{false};
  bool missing_owner{false};
  bool duplicate_owner{false};
  bool missing_ghost_payload{false};
  std::vector<std::string> messages{};
};

struct CutDistributedState {
  CutDistributedRevisionSnapshot revision{};
  CutDistributedExchangePacket local_packet{};
  CutDistributedExchangePacket exchanged_packet{};
  bool neighbor_sparse_exchange{false};
  std::vector<rank_t> communication_neighbors{};
  std::vector<rank_t> received_neighbor_ranks{};
  std::vector<CutDistributedEntityRecord> owned_records{};
  std::vector<CutDistributedEntityRecord> ghost_records{};
  std::vector<CutDistributedEntityRecord> imported_records{};
  std::vector<std::string> diagnostics{};

  [[nodiscard]] bool valid_for(const DistributedMesh& mesh,
                               const CutClassificationMap& map,
                               const CutTopologyRecord& topology) const noexcept;
};

struct CutSupportMatrixEntry {
  CellFamily parent_family{CellFamily::Point};
  int geometry_order{1};
  EmbeddedGeometryKind embedded_kind{EmbeddedGeometryKind::Plane};
  bool distributed{false};
  std::string cut_mode{"linearized-cut"};
  std::string quadrature_policy{"topology-subdivision"};
  std::string conditioning_policy{"geometric-conditioning-hooks"};
  std::string fe_execution_path{"shared-cut-integration-record"};
  CutSupportStatus status{CutSupportStatus::Unsupported};
  std::string qualification{};
};

enum class CutSupportEvidenceDomain : std::uint8_t {
  Topology,
  Quadrature,
  FeExecution,
  RestartRollback,
  Mpi,
  Sensitivity,
  Diagnostic,
  FullValidation
};

enum class CutSupportAuditCategory : std::uint8_t {
  Unsupported,
  FullyValidated,
  MissingAnalyticValidation,
  MissingSensitivityEvidence,
  MissingRestartOrMpiEvidence,
  MissingFeExecutionEvidence,
  AdvertisedTooBroadly
};

struct CutSupportMatrixEvidenceRecord {
  CutSupportMatrixEntry entry{};
  CutSupportEvidenceDomain domain{CutSupportEvidenceDomain::Topology};
  std::string evidence_id{};
  std::string description{};
  std::string verification{};
};

struct CutSupportMatrixQualificationRecord {
  CutSupportMatrixEntry entry{};
  bool requires_topology_evidence{false};
  bool requires_quadrature_evidence{false};
  bool requires_fe_execution_evidence{false};
  bool requires_restart_rollback_evidence{false};
  bool requires_mpi_evidence{false};
  bool requires_sensitivity_evidence{false};
  bool requires_diagnostic_evidence{false};
  bool requires_validation_evidence{false};
  bool topology_evidence{false};
  bool quadrature_evidence{false};
  bool fe_execution_evidence{false};
  bool restart_rollback_evidence{false};
  bool mpi_evidence{false};
  bool sensitivity_evidence{false};
  bool diagnostic_evidence{false};
  bool validation_evidence{false};
  bool qualified{false};
  std::vector<std::string> evidence{};
  std::vector<std::string> missing{};
};

struct CutSupportMatrixAuditRecord {
  CutSupportMatrixQualificationRecord qualification{};
  CutSupportAuditCategory category{CutSupportAuditCategory::Unsupported};
  std::vector<CutSupportEvidenceDomain> missing_domains{};
  std::string summary{};
};

class EmbeddedGeometryRegistry {
public:
  void register_geometry(EmbeddedGeometryDescriptor descriptor);
  [[nodiscard]] bool contains(const std::string& persistent_id) const noexcept;
  [[nodiscard]] const EmbeddedGeometryDescriptor* find(const std::string& persistent_id) const noexcept;
  [[nodiscard]] const EmbeddedGeometryDescriptor& require(const std::string& persistent_id) const;
  void erase(const std::string& persistent_id);
  void clear();

  [[nodiscard]] std::vector<std::string> active_geometry_ids() const;
  [[nodiscard]] std::vector<CutClassificationMap> classify_active(
      const MeshBase& mesh,
      const CutClassificationOptions& options = {}) const;
  [[nodiscard]] EmbeddedGeometryRegistrySnapshot snapshot() const noexcept;
  [[nodiscard]] std::vector<EmbeddedGeometryDescriptor> descriptors() const;

private:
  std::unordered_map<std::string, EmbeddedGeometryDescriptor> geometries_{};
  std::uint64_t registry_epoch_{0};
};

class CutClassificationTransaction {
public:
  explicit CutClassificationTransaction(CutClassificationMap& map);

  CutClassificationTransaction(const CutClassificationTransaction&) = delete;
  CutClassificationTransaction& operator=(const CutClassificationTransaction&) = delete;

  void stage(CutClassificationMap next);
  void accept();
  void rollback();

  [[nodiscard]] CutClassificationState state() const noexcept { return state_; }

private:
  CutClassificationMap* map_{nullptr};
  CutClassificationMap backup_{};
  CutClassificationState state_{CutClassificationState::Trial};
};

[[nodiscard]] CutClassification classify_signed_distances(
    const std::vector<real_t>& signed_distances,
    real_t tolerance) noexcept;

[[nodiscard]] CutClassificationMap classify_embedded_geometry(
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& embedded_geometry,
    const CutClassificationOptions& options = {});

[[nodiscard]] CutClassificationMap classify_embedded_geometry(
    const DistributedMesh& mesh,
    const EmbeddedGeometryDescriptor& embedded_geometry,
    const CutClassificationOptions& options = {});

[[nodiscard]] EmbeddedGeometryQueryDiagnostic diagnose_embedded_geometry_query_support(
    const EmbeddedGeometryDescriptor& embedded_geometry);

[[nodiscard]] EmbeddedGeometryDescriptor make_triangulated_surface_descriptor(
    std::string persistent_id,
    std::vector<EmbeddedSurfaceTriangle> triangles,
    Configuration configuration = Configuration::Reference,
    std::uint64_t source_surface_revision = 1);

[[nodiscard]] EmbeddedGeometryDescriptor read_ascii_stl_embedded_surface(
    const std::string& path,
    std::string persistent_id,
    Configuration configuration = Configuration::Reference,
    std::uint64_t source_surface_revision = 1);

[[nodiscard]] std::vector<EmbeddedGeometryRestartRecord> make_embedded_geometry_restart_records(
    const EmbeddedGeometryRegistry& registry);

[[nodiscard]] EmbeddedGeometryRegistry restore_embedded_geometry_registry(
    const std::vector<EmbeddedGeometryRestartRecord>& records);

[[nodiscard]] CutOperationDiagnostic diagnose_cut_operation(
    const CutClassificationMap& map,
    std::string operation);

[[nodiscard]] EmbeddedGeometryQueryDiagnostic diagnose_boolean_region_composition(
    const EmbeddedGeometryDescriptor& embedded_geometry,
    const std::vector<std::array<real_t, 3>>& sample_points,
    const CutPredicatePolicy& predicate_policy = {});

[[nodiscard]] CutTopologyRecord reconstruct_cut_topology(
    const MeshBase& mesh,
    const CutClassificationMap& map,
    const CutTopologyOptions& options = {});

[[nodiscard]] CutCurvedValidityDiagnostic diagnose_cut_topology_validity(
    const CutTopologyRecord& topology,
    bool high_order_parent_geometry = false,
    const CutCurvedValidityPolicy& policy = {});

[[nodiscard]] CutTopologyRecord project_cut_topology_to_embedded_geometry(
    const CutTopologyRecord& topology,
    const EmbeddedGeometryDescriptor& embedded_geometry,
    std::uint64_t new_topology_revision_salt = 0);

[[nodiscard]] CutDistributedExchangePacket make_distributed_cut_exchange_packet(
    const CutClassificationMap& map,
    const CutTopologyRecord& topology);

[[nodiscard]] CutDistributedExchangePacket deduplicate_cut_exchange_packet(
    CutDistributedExchangePacket packet);

[[nodiscard]] CutDistributedState build_distributed_cut_state(
    const DistributedMesh& mesh,
    const CutClassificationMap& map,
    const CutTopologyRecord& topology);

[[nodiscard]] CutDistributedStateDiagnostic diagnose_distributed_cut_state(
    const CutDistributedState& state);

[[nodiscard]] std::vector<CutSupportMatrixEntry> cut_support_matrix();

[[nodiscard]] std::vector<CutSupportMatrixEvidenceRecord>
cut_support_matrix_validation_ledger();

[[nodiscard]] std::vector<CutSupportMatrixQualificationRecord>
qualify_cut_support_matrix();

[[nodiscard]] std::vector<CutSupportMatrixAuditRecord>
audit_cut_support_matrix_validation();

[[nodiscard]] CutSupportMatrixEntry evaluate_cut_support(
    CellFamily parent_family,
    int geometry_order,
    EmbeddedGeometryKind embedded_kind,
    bool distributed,
    const std::string& cut_mode,
    const std::string& quadrature_policy);

[[nodiscard]] CutSupportMatrixEntry evaluate_cut_support(
    CellFamily parent_family,
    int geometry_order,
    EmbeddedGeometryKind embedded_kind,
    bool distributed,
    const std::string& cut_mode,
    const std::string& quadrature_policy,
    const std::string& conditioning_policy,
    const std::string& fe_execution_path);

[[nodiscard]] CutClassificationRestartRecord make_cut_classification_restart_record(
    const CutClassificationMap& map);

[[nodiscard]] CutClassificationRestartRecord make_cut_classification_restart_record(
    const CutClassificationMap& map,
    const CutTopologyRecord& topology);

} // namespace search
} // namespace svmp

#endif // SVMP_SEARCH_CUT_CELL_H
