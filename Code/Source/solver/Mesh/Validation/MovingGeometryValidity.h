/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_MOVING_GEOMETRY_VALIDITY_H
#define SVMP_MOVING_GEOMETRY_VALIDITY_H

#include "../Core/MeshTypes.h"
#include "../Observer/MeshObserver.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace svmp {

class MeshBase;
class DistributedMesh;

namespace validation {

enum class ValiditySeverity {
  Info,
  Warning,
  Error
};

enum class ValidityAction {
  Warn,
  Reject,
  Backtrack,
  Constrain
};

enum class MovingGeometryCheck {
  DegenerateBoundary,
  BoundarySelfIntersection,
  SurfaceFolding,
  NormalOrientation,
  MinimumSeparation,
  ContactSeparation,
  ShellThicknessSeparation,
  CurvedBoundarySampling,
  SweptVolume,
  BoundaryLayer,
  TwoDBoundary,
  ManifoldConstraint,
  DirectionalConstraint,
  ActiveInequality
};

enum class ValidityPolicyGroup {
  Custom,
  ALEBasic,
  Contact,
  FreeSurface,
  Shell,
  BoundaryLayer,
  LargeStep
};

enum class MotionConstraintKind {
  ManifoldPlane,
  NormalOnly,
  TangentialOnly,
  SurfaceSliding,
  BoundaryLayerPreservation,
  MaximumDisplacement,
  MinimumSeparation
};

struct RobustPredicatePolicy {
  real_t intersection_tolerance = 1.0e-10;
  real_t near_contact_tolerance = 1.0e-8;
  real_t coplanar_tolerance = 1.0e-10;
  real_t degenerate_tolerance = 1.0e-12;
  real_t curved_sampling_tolerance = 1.0e-10;
  real_t nonfinite_tolerance = 0.0;
  real_t aabb_padding = 1.0e-12;
};

struct LabelPairScope {
  label_t first = INVALID_LABEL;
  label_t second = INVALID_LABEL;
  bool symmetric = true;

  [[nodiscard]] bool matches(label_t a, label_t b) const noexcept;
};

struct MovingGeometryCheckSpec {
  MovingGeometryCheck check = MovingGeometryCheck::DegenerateBoundary;
  bool enabled = true;
  std::string name;
  ValiditySeverity severity = ValiditySeverity::Error;
  ValidityAction action = ValidityAction::Reject;
  real_t threshold = 0.0;
  std::set<label_t> labels;
  std::vector<LabelPairScope> label_pairs;
  bool physics_neutral_constraint_output = false;
};

struct MotionConstraintSpec {
  MotionConstraintKind kind = MotionConstraintKind::ManifoldPlane;
  std::string name;
  label_t label = INVALID_LABEL;
  LabelPairScope label_pair{};
  std::array<real_t, 3> point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> direction{{0.0, 0.0, 1.0}};
  real_t threshold = 1.0e-10;
  ValidityAction action = ValidityAction::Backtrack;
  bool physics_neutral = true;
};

struct MovingGeometryValidityPolicy {
  ValidityPolicyGroup group = ValidityPolicyGroup::Custom;
  std::string group_name = "Custom";
  RobustPredicatePolicy robust{};
  Configuration configuration = Configuration::Current;
  real_t time_level = 0.0;
  std::vector<MovingGeometryCheckSpec> checks;
  std::vector<MotionConstraintSpec> constraints;

  [[nodiscard]] bool enables(MovingGeometryCheck check) const noexcept;
  [[nodiscard]] const MovingGeometryCheckSpec* spec(MovingGeometryCheck check) const noexcept;
  [[nodiscard]] std::map<std::string, std::string> restart_metadata() const;
};

struct MovingGeometryValidityFailure {
  std::string check_name;
  std::string message;
  ValiditySeverity severity = ValiditySeverity::Error;
  ValidityAction recommended_action = ValidityAction::Reject;
  EntityKind entity_kind = EntityKind::Face;
  std::vector<index_t> local_ids;
  std::vector<gid_t> global_ids;
  std::vector<label_t> labels;
  real_t measured_value = 0.0;
  real_t threshold = 0.0;
  Configuration configuration = Configuration::Current;
  MeshRevisionState revision_state{};
  real_t time_level = 0.0;
  int owner_rank = 0;
  std::uint64_t canonical_key = 0;
};

struct MovingGeometryValidityReport {
  bool passed = true;
  std::string policy_group_name = "Custom";
  Configuration configuration = Configuration::Current;
  MeshRevisionState revision_state{};
  real_t time_level = 0.0;
  std::vector<MovingGeometryValidityFailure> failures;
  std::size_t broad_phase_candidate_pairs = 0;
  std::size_t exact_candidate_pairs = 0;

  void add_failure(MovingGeometryValidityFailure failure);
  [[nodiscard]] bool has_failures() const noexcept { return !failures.empty(); }
  [[nodiscard]] bool requires_rejection() const noexcept;
  [[nodiscard]] bool recommends_backtrack() const noexcept;
  [[nodiscard]] bool provides_constraints() const noexcept;
  [[nodiscard]] std::string to_string() const;
  [[nodiscard]] std::map<std::string, std::string> restart_metadata() const;
};

class MovingGeometryValidity {
public:
  static MovingGeometryValidityPolicy preset(ValidityPolicyGroup group);
  static MovingGeometryValidityPolicy ale_basic_policy();
  static MovingGeometryValidityPolicy contact_policy();
  static MovingGeometryValidityPolicy free_surface_policy();
  static MovingGeometryValidityPolicy shell_policy();
  static MovingGeometryValidityPolicy boundary_layer_policy();
  static MovingGeometryValidityPolicy large_step_policy();

  static MovingGeometryValidityReport evaluate(const MeshBase& mesh,
                                               const MovingGeometryValidityPolicy& policy);
  static MovingGeometryValidityReport evaluate(const DistributedMesh& mesh,
                                               const MovingGeometryValidityPolicy& policy);

  static const char* check_name(MovingGeometryCheck check) noexcept;
  static const char* action_name(ValidityAction action) noexcept;
  static const char* severity_name(ValiditySeverity severity) noexcept;
  static const char* policy_group_name(ValidityPolicyGroup group) noexcept;
};

} // namespace validation
} // namespace svmp

#endif // SVMP_MOVING_GEOMETRY_VALIDITY_H
