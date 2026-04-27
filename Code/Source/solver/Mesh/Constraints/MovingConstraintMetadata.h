/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_MESH_CONSTRAINTS_MOVING_CONSTRAINT_METADATA_H
#define SVMP_MESH_CONSTRAINTS_MOVING_CONSTRAINT_METADATA_H

#include "../Core/MeshTypes.h"
#include "../Observer/MeshObserver.h"
#include "../Motion/IMotionBackend.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace svmp {

class MeshBase;

namespace constraints {

enum class MovingConstraintKind : std::uint8_t {
  PeriodicBoundary,
  TiedBoundary,
  GeometricContinuity
};

struct MeshConstraintRevisionDependencies {
  bool geometry{false};
  bool topology{false};
  bool ownership{false};
  bool numbering{false};
  bool field_layout{false};
  bool labels{false};
  bool active_configuration{false};

  [[nodiscard]] bool any() const noexcept;

  [[nodiscard]] static MeshConstraintRevisionDependencies geometry_only() noexcept;
  [[nodiscard]] static MeshConstraintRevisionDependencies topology_labels() noexcept;
  [[nodiscard]] static MeshConstraintRevisionDependencies moving_boundary_relation() noexcept;
};

struct MeshConstraintRevisionSnapshot {
  bool valid{false};
  std::uint64_t geometry{0};
  std::uint64_t topology{0};
  std::uint64_t ownership{0};
  std::uint64_t numbering{0};
  std::uint64_t field_layout{0};
  std::uint64_t labels{0};
  std::uint64_t active_configuration{0};

  [[nodiscard]] static MeshConstraintRevisionSnapshot capture(const MeshBase& mesh);
};

[[nodiscard]] bool dependency_changed(
    const MeshConstraintRevisionDependencies& deps,
    const MeshConstraintRevisionSnapshot& cached,
    const MeshConstraintRevisionSnapshot& current) noexcept;

struct PeriodicBoundaryMotionMetadata {
  label_t slave_label{INVALID_LABEL};
  label_t master_label{INVALID_LABEL};
  std::array<real_t, 3> slave_to_master_translation{{0.0, 0.0, 0.0}};
  real_t matching_tolerance{1e-10};
  bool preserve_relative_motion{true};
};

struct TiedBoundaryMotionMetadata {
  label_t slave_label{INVALID_LABEL};
  label_t master_label{INVALID_LABEL};
  real_t max_pair_distance{std::numeric_limits<real_t>::infinity()};
  bool update_relation_map_on_motion{true};
};

struct GeometricContinuityMetadata {
  label_t boundary_label{INVALID_LABEL};
  int geometry_order{1};
  bool enforce_curved_geometry_dofs{true};
};

struct MovingMeshConstraintMetadata {
  std::string name;
  MovingConstraintKind kind{MovingConstraintKind::GeometricContinuity};
  MeshConstraintRevisionDependencies dependencies{};
  PeriodicBoundaryMotionMetadata periodic{};
  TiedBoundaryMotionMetadata tied{};
  GeometricContinuityMetadata continuity{};
};

struct MotionConstraintValidationResult {
  bool ok{true};
  std::string message;
  std::vector<std::string> diagnostics;
};

class MovingMeshConstraintRegistry {
public:
  void clear();
  void add(MovingMeshConstraintMetadata metadata);

  [[nodiscard]] bool empty() const noexcept { return metadata_.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return metadata_.size(); }
  [[nodiscard]] const std::vector<MovingMeshConstraintMetadata>& constraints() const noexcept {
    return metadata_;
  }

  [[nodiscard]] MeshConstraintRevisionDependencies combined_dependencies() const noexcept;
  [[nodiscard]] MotionConstraintValidationResult validate_prescribed_motion(
      const MeshBase& mesh,
      const std::vector<motion::MotionDirichletBC>& bcs,
      double dt,
      double step_scale,
      real_t tolerance = 1e-10) const;

private:
  std::vector<MovingMeshConstraintMetadata> metadata_{};
};

[[nodiscard]] MovingMeshConstraintMetadata make_periodic_boundary_metadata(
    std::string name,
    label_t slave_label,
    label_t master_label,
    std::array<real_t, 3> slave_to_master_translation = {{0.0, 0.0, 0.0}});

[[nodiscard]] MovingMeshConstraintMetadata make_tied_boundary_metadata(
    std::string name,
    label_t slave_label,
    label_t master_label);

[[nodiscard]] MovingMeshConstraintMetadata make_geometric_continuity_metadata(
    std::string name,
    label_t boundary_label,
    int geometry_order = 1);

} // namespace constraints
} // namespace svmp

#endif // SVMP_MESH_CONSTRAINTS_MOVING_CONSTRAINT_METADATA_H
