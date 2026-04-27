/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_MOTION_MAP_H
#define SVMP_MOTION_MAP_H

/**
 * @file MotionMap.h
 * @brief Physics-agnostic prescribed motion maps for moving and rotating meshes.
 *
 * Motion maps are Mesh-library infrastructure only.  They define how geometry
 * DOFs move in space, publish mesh-motion fields, and provide transaction /
 * restart metadata for higher-level solvers.  They do not encode any fluid,
 * structure, or interface physics.
 */

#include "../Core/MeshTypes.h"
#include "MotionFields.h"
#include "MotionState.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace svmp {

class MeshBase;

namespace motion {

enum class MotionMapKind : std::uint8_t {
  RigidBody,
  Affine,
  UserDefined
};

enum class MotionMapTargetKind : std::uint8_t {
  AllGeometryDofs,
  ExplicitGeometryDofs,
  BoundaryLabel,
  RegionLabel,
  VertexLabel
};

enum class MotionMapTransactionState : std::uint8_t {
  Empty,
  Trial,
  Accepted,
  RolledBack
};

enum class MotionMapTimeLevel : std::uint8_t {
  TrialIterate,
  AcceptedNonlinearState,
  AcceptedTimeStep,
  AcceptedRemeshOrRezoneState
};

struct MotionMapTarget {
  MotionMapTargetKind kind{MotionMapTargetKind::AllGeometryDofs};
  label_t label{INVALID_LABEL};
  std::vector<index_t> geometry_dofs{};
  std::string logical_region_id{};
  std::string physical_label_name{};

  [[nodiscard]] static MotionMapTarget all(std::string logical_region_id = {});
  [[nodiscard]] static MotionMapTarget explicit_dofs(std::vector<index_t> dofs,
                                                     std::string logical_region_id = {});
  [[nodiscard]] static MotionMapTarget boundary(label_t label,
                                                std::string logical_region_id = {});
  [[nodiscard]] static MotionMapTarget region(label_t label,
                                              std::string logical_region_id = {});
  [[nodiscard]] static MotionMapTarget vertex_label(label_t label,
                                                    std::string logical_region_id = {});
};

struct MotionMapTimeState {
  real_t time{0.0};
  real_t reference_time{0.0};
  real_t dt{0.0};
  MotionMapTimeLevel time_level{MotionMapTimeLevel::TrialIterate};
};

struct MotionMapPointState {
  std::array<real_t, 3> reference_point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> previous_point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> current_point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> displacement{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> velocity{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> acceleration{{0.0, 0.0, 0.0}};
};

class IMotionMap {
public:
  virtual ~IMotionMap() = default;

  [[nodiscard]] virtual MotionMapKind kind() const noexcept = 0;
  [[nodiscard]] virtual const std::string& name() const noexcept = 0;
  [[nodiscard]] virtual MotionMapPointState evaluate(
      const std::array<real_t, 3>& reference_point,
      const std::array<real_t, 3>& previous_point,
      const MotionMapTimeState& time_state) const = 0;
};

struct RigidBodyMotionParameters {
  std::array<real_t, 3> origin{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> rotation_axis{{0.0, 0.0, 1.0}};
  real_t initial_angle{0.0};
  real_t angular_speed{0.0};
  real_t angular_acceleration{0.0};
  std::array<real_t, 3> initial_translation{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> linear_velocity{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> linear_acceleration{{0.0, 0.0, 0.0}};
};

class RigidBodyMotionMap final : public IMotionMap {
public:
  explicit RigidBodyMotionMap(RigidBodyMotionParameters parameters,
                              std::string name = "rigid_body_motion");

  [[nodiscard]] MotionMapKind kind() const noexcept override { return MotionMapKind::RigidBody; }
  [[nodiscard]] const std::string& name() const noexcept override { return name_; }
  [[nodiscard]] const RigidBodyMotionParameters& parameters() const noexcept { return parameters_; }

  [[nodiscard]] MotionMapPointState evaluate(
      const std::array<real_t, 3>& reference_point,
      const std::array<real_t, 3>& previous_point,
      const MotionMapTimeState& time_state) const override;

private:
  RigidBodyMotionParameters parameters_{};
  std::string name_{};
};

struct AffineMotionParameters {
  std::array<real_t, 3> origin{{0.0, 0.0, 0.0}};
  std::array<std::array<real_t, 3>, 3> transform{{
      {{1.0, 0.0, 0.0}},
      {{0.0, 1.0, 0.0}},
      {{0.0, 0.0, 1.0}}}};
  std::array<std::array<real_t, 3>, 3> velocity_gradient{};
  std::array<std::array<real_t, 3>, 3> acceleration_gradient{};
  std::array<real_t, 3> translation{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> linear_velocity{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> linear_acceleration{{0.0, 0.0, 0.0}};
};

class AffineMotionMap final : public IMotionMap {
public:
  explicit AffineMotionMap(AffineMotionParameters parameters,
                           std::string name = "affine_motion");

  [[nodiscard]] MotionMapKind kind() const noexcept override { return MotionMapKind::Affine; }
  [[nodiscard]] const std::string& name() const noexcept override { return name_; }
  [[nodiscard]] const AffineMotionParameters& parameters() const noexcept { return parameters_; }

  [[nodiscard]] MotionMapPointState evaluate(
      const std::array<real_t, 3>& reference_point,
      const std::array<real_t, 3>& previous_point,
      const MotionMapTimeState& time_state) const override;

private:
  AffineMotionParameters parameters_{};
  std::string name_{};
};

struct MotionMapApplyOptions {
  bool update_current_coordinates{true};
  bool update_motion_fields{true};
  bool update_previous_coordinates{true};
  bool set_active_configuration_current{true};
  bool require_finite_values{true};
  bool allow_empty_target{false};
};

struct MotionMapApplyResult {
  MotionMapKind kind{MotionMapKind::UserDefined};
  MotionMapTarget target{};
  MotionMapTimeState time_state{};
  std::vector<index_t> geometry_dofs{};
  std::vector<MotionMapPointState> dof_states{};
  std::uint64_t geometry_revision_before{0};
  std::uint64_t geometry_revision_after{0};
  std::uint64_t active_configuration_epoch_before{0};
  std::uint64_t active_configuration_epoch_after{0};
  MotionMapTransactionState transaction_state{MotionMapTransactionState::Trial};
};

struct MotionMapRestartRecord {
  std::string map_name{};
  MotionMapKind map_kind{MotionMapKind::UserDefined};
  MotionMapTarget target{};
  MotionMapTimeState time_state{};
  std::uint64_t geometry_revision{0};
  std::uint64_t topology_revision{0};
  std::uint64_t ownership_revision{0};
  std::uint64_t numbering_revision{0};
  std::uint64_t field_layout_revision{0};
  std::uint64_t label_revision{0};
  std::uint64_t active_configuration_epoch{0};
};

class MotionMapTransaction {
public:
  explicit MotionMapTransaction(MeshBase& mesh);

  MotionMapTransaction(const MotionMapTransaction&) = delete;
  MotionMapTransaction& operator=(const MotionMapTransaction&) = delete;

  [[nodiscard]] MotionMapTransactionState state() const noexcept { return state_; }
  [[nodiscard]] std::uint64_t entry_geometry_revision() const noexcept {
    return entry_geometry_revision_;
  }

  MotionMapApplyResult apply(const IMotionMap& motion_map,
                             const MotionMapTarget& target,
                             const MotionMapTimeState& time_state,
                             const MotionMapApplyOptions& options = {});
  void accept();
  void rollback();

private:
  MeshBase* mesh_{nullptr};
  MotionCoordinateBackup backup_{};
  std::uint64_t entry_geometry_revision_{0};
  MotionMapTransactionState state_{MotionMapTransactionState::Empty};
};

[[nodiscard]] std::vector<index_t> select_motion_map_geometry_dofs(
    const MeshBase& mesh,
    const MotionMapTarget& target);

MotionMapApplyResult apply_motion_map(MeshBase& mesh,
                                      const IMotionMap& motion_map,
                                      const MotionMapTarget& target,
                                      const MotionMapTimeState& time_state,
                                      const MotionMapApplyOptions& options = {});

MotionMapApplyResult apply_motion_map(Mesh& mesh,
                                      const IMotionMap& motion_map,
                                      const MotionMapTarget& target,
                                      const MotionMapTimeState& time_state,
                                      const MotionMapApplyOptions& options = {});

[[nodiscard]] MotionMapRestartRecord make_motion_map_restart_record(
    const MeshBase& mesh,
    const IMotionMap& motion_map,
    const MotionMapTarget& target,
    const MotionMapTimeState& time_state);

} // namespace motion
} // namespace svmp

#endif // SVMP_MOTION_MAP_H
