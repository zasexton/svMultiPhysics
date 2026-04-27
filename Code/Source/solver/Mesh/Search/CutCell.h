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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace svmp {

class MeshBase;
class DistributedMesh;

namespace search {

enum class EmbeddedGeometryKind : std::uint8_t {
  Plane,
  Sphere
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

struct EmbeddedRegionProvenance {
  std::string persistent_id{};
  std::string name{};
  label_t physical_label{INVALID_LABEL};
  std::uint64_t provenance_epoch{0};

  [[nodiscard]] bool empty() const noexcept { return persistent_id.empty() && name.empty(); }
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
  std::uint64_t embedded_constraint_epoch{0};
  std::uint64_t fe_layout_revision{0};

  [[nodiscard]] static EmbeddedRevisionSnapshot capture(
      const MeshBase& mesh,
      Configuration configuration,
      std::uint64_t embedded_geometry_epoch,
      std::uint64_t embedded_constraint_epoch = 0,
      std::uint64_t fe_layout_revision = 0);

  [[nodiscard]] bool matches(const MeshBase& mesh,
                             Configuration configuration,
                             std::uint64_t embedded_geometry_epoch,
                             std::uint64_t embedded_constraint_epoch = 0,
                             std::uint64_t fe_layout_revision = 0) const noexcept;
  [[nodiscard]] std::uint64_t revision_key() const noexcept;
};

struct EmbeddedGeometryDescriptor {
  EmbeddedGeometryKind kind{EmbeddedGeometryKind::Plane};
  Configuration configuration{Configuration::Reference};
  std::array<real_t, 3> origin{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> normal{{1.0, 0.0, 0.0}};
  real_t radius{1.0};
  std::uint64_t geometry_epoch{0};
  EmbeddedRegionProvenance provenance{};

  [[nodiscard]] real_t signed_distance(const std::array<real_t, 3>& point) const noexcept;
  [[nodiscard]] std::array<real_t, 3> outward_normal(const std::array<real_t, 3>& point) const noexcept;
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
};

struct CutClassificationOptions {
  Configuration configuration{Configuration::Reference};
  real_t tolerance{1.0e-12};
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

  [[nodiscard]] bool valid_for(const MeshBase& mesh) const noexcept;
  [[nodiscard]] std::uint64_t revision_key() const noexcept;
  void accept_trial() noexcept;
  void rollback_trial();
};

struct CutClassificationRestartRecord {
  std::string name{};
  EmbeddedRegionProvenance provenance{};
  EmbeddedGeometryKind embedded_kind{EmbeddedGeometryKind::Plane};
  std::uint64_t revision_key{0};
  std::uint64_t embedded_geometry_epoch{0};
  std::uint64_t embedded_constraint_epoch{0};
  std::uint64_t fe_layout_revision{0};
  std::size_t cut_cell_count{0};
  std::size_t cut_face_count{0};
  std::size_t cut_edge_count{0};
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

[[nodiscard]] CutClassificationRestartRecord make_cut_classification_restart_record(
    const CutClassificationMap& map);

} // namespace search
} // namespace svmp

#endif // SVMP_SEARCH_CUT_CELL_H
