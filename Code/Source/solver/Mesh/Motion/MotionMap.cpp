/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "MotionMap.h"

#include "../Core/DistributedMesh.h"
#include "../Core/MeshBase.h"
#include "../Fields/MeshFields.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace svmp {
namespace motion {
namespace {

std::array<real_t, 3> add(const std::array<real_t, 3>& a,
                          const std::array<real_t, 3>& b) noexcept {
  return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
}

std::array<real_t, 3> sub(const std::array<real_t, 3>& a,
                          const std::array<real_t, 3>& b) noexcept {
  return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

std::array<real_t, 3> scale(const std::array<real_t, 3>& a, real_t s) noexcept {
  return {{s * a[0], s * a[1], s * a[2]}};
}

real_t dot(const std::array<real_t, 3>& a,
           const std::array<real_t, 3>& b) noexcept {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<real_t, 3> cross(const std::array<real_t, 3>& a,
                            const std::array<real_t, 3>& b) noexcept {
  return {{a[1] * b[2] - a[2] * b[1],
           a[2] * b[0] - a[0] * b[2],
           a[0] * b[1] - a[1] * b[0]}};
}

real_t norm(const std::array<real_t, 3>& a) noexcept {
  return std::sqrt(dot(a, a));
}

std::array<real_t, 3> unit_or_default(std::array<real_t, 3> axis) noexcept {
  const real_t n = norm(axis);
  if (n <= real_t{1.0e-30}) {
    return {{0.0, 0.0, 1.0}};
  }
  return scale(axis, real_t{1.0} / n);
}

std::array<real_t, 3> rotate_rodrigues(const std::array<real_t, 3>& r,
                                       const std::array<real_t, 3>& axis_unit,
                                       real_t angle) noexcept {
  const real_t c = std::cos(angle);
  const real_t s = std::sin(angle);
  return add(add(scale(r, c), scale(cross(axis_unit, r), s)),
             scale(axis_unit, dot(axis_unit, r) * (real_t{1.0} - c)));
}

std::array<real_t, 3> matvec(const std::array<std::array<real_t, 3>, 3>& a,
                             const std::array<real_t, 3>& x) noexcept {
  return {{a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
           a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
           a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2]}};
}

bool finite_vec(const std::array<real_t, 3>& v) noexcept {
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

void append_unique(std::vector<index_t>& dofs, index_t dof) {
  if (dof != INVALID_INDEX) {
    dofs.push_back(dof);
  }
}

void append_face_geometry_dofs(const MeshBase& mesh,
                               index_t face,
                               std::vector<index_t>& dofs) {
  const auto span = mesh.face_vertices_span(face);
  for (size_t i = 0; i < span.second; ++i) {
    append_unique(dofs, span.first[i]);
  }
}

void append_cell_geometry_dofs(const MeshBase& mesh,
                               index_t cell,
                               std::vector<index_t>& dofs) {
  const auto span = mesh.cell_vertices_span(cell);
  for (size_t i = 0; i < span.second; ++i) {
    append_unique(dofs, span.first[i]);
  }
}

void sort_unique_validate(const MeshBase& mesh, std::vector<index_t>& dofs) {
  std::sort(dofs.begin(), dofs.end());
  dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
  for (const index_t dof : dofs) {
    if (dof < 0 || static_cast<size_t>(dof) >= mesh.geometry_dof_count()) {
      throw std::out_of_range("motion-map target references an invalid geometry DOF");
    }
  }
}

void publish_motion_fields(MeshBase& mesh,
                           const MotionMapApplyResult& result,
                           const MotionMapApplyOptions& options) {
  if (!options.update_motion_fields || result.geometry_dofs.empty()) {
    return;
  }

  const int dim = mesh.dim();
  auto handles = attach_motion_fields(mesh, dim);

  real_t* displacement = MeshFields::field_data_as<real_t>(mesh, handles.displacement);
  real_t* velocity = MeshFields::field_data_as<real_t>(mesh, handles.velocity);
  real_t* acceleration = MeshFields::field_data_as<real_t>(mesh, handles.acceleration);
  real_t* previous_coordinates =
      MeshFields::field_data_as<real_t>(mesh, handles.previous_coordinates);

  const size_t components = static_cast<size_t>(dim);
  for (size_t i = 0; i < result.geometry_dofs.size(); ++i) {
    const auto dof = result.geometry_dofs[i];
    const auto& state = result.dof_states[i];
    const size_t base = static_cast<size_t>(dof) * components;
    for (int d = 0; d < dim; ++d) {
      const size_t k = static_cast<size_t>(d);
      displacement[base + k] = state.displacement[k];
      velocity[base + k] = state.velocity[k];
      acceleration[base + k] = state.acceleration[k];
      if (options.update_previous_coordinates) {
        previous_coordinates[base + k] = state.previous_point[k];
      }
    }
  }
}

} // namespace

MotionMapTarget MotionMapTarget::all(std::string logical_region_id) {
  MotionMapTarget target;
  target.kind = MotionMapTargetKind::AllGeometryDofs;
  target.logical_region_id = std::move(logical_region_id);
  return target;
}

MotionMapTarget MotionMapTarget::explicit_dofs(std::vector<index_t> dofs,
                                               std::string logical_region_id) {
  MotionMapTarget target;
  target.kind = MotionMapTargetKind::ExplicitGeometryDofs;
  target.geometry_dofs = std::move(dofs);
  target.logical_region_id = std::move(logical_region_id);
  return target;
}

MotionMapTarget MotionMapTarget::boundary(label_t label, std::string logical_region_id) {
  MotionMapTarget target;
  target.kind = MotionMapTargetKind::BoundaryLabel;
  target.label = label;
  target.logical_region_id = std::move(logical_region_id);
  return target;
}

MotionMapTarget MotionMapTarget::region(label_t label, std::string logical_region_id) {
  MotionMapTarget target;
  target.kind = MotionMapTargetKind::RegionLabel;
  target.label = label;
  target.logical_region_id = std::move(logical_region_id);
  return target;
}

MotionMapTarget MotionMapTarget::vertex_label(label_t label, std::string logical_region_id) {
  MotionMapTarget target;
  target.kind = MotionMapTargetKind::VertexLabel;
  target.label = label;
  target.logical_region_id = std::move(logical_region_id);
  return target;
}

RigidBodyMotionMap::RigidBodyMotionMap(RigidBodyMotionParameters parameters, std::string name)
    : parameters_(std::move(parameters))
    , name_(std::move(name)) {}

MotionMapPointState RigidBodyMotionMap::evaluate(
    const std::array<real_t, 3>& reference_point,
    const std::array<real_t, 3>& previous_point,
    const MotionMapTimeState& time_state) const {
  const real_t tau = time_state.time - time_state.reference_time;
  const auto axis = unit_or_default(parameters_.rotation_axis);
  const real_t theta = parameters_.initial_angle +
                       parameters_.angular_speed * tau +
                       real_t{0.5} * parameters_.angular_acceleration * tau * tau;
  const real_t speed = parameters_.angular_speed +
                       parameters_.angular_acceleration * tau;
  const auto omega = scale(axis, speed);
  const auto alpha = scale(axis, parameters_.angular_acceleration);
  const auto translation =
      add(parameters_.initial_translation,
          add(scale(parameters_.linear_velocity, tau),
              scale(parameters_.linear_acceleration, real_t{0.5} * tau * tau)));

  const auto r0 = sub(reference_point, parameters_.origin);
  const auto rotated_r = rotate_rodrigues(r0, axis, theta);
  const auto current = add(add(parameters_.origin, rotated_r), translation);
  const auto velocity = add(add(parameters_.linear_velocity,
                                scale(parameters_.linear_acceleration, tau)),
                            cross(omega, rotated_r));
  const auto acceleration = add(parameters_.linear_acceleration,
                                add(cross(alpha, rotated_r),
                                    cross(omega, cross(omega, rotated_r))));

  MotionMapPointState state;
  state.reference_point = reference_point;
  state.previous_point = previous_point;
  state.current_point = current;
  state.displacement = sub(current, reference_point);
  state.velocity = velocity;
  state.acceleration = acceleration;
  return state;
}

AffineMotionMap::AffineMotionMap(AffineMotionParameters parameters, std::string name)
    : parameters_(std::move(parameters))
    , name_(std::move(name)) {}

MotionMapPointState AffineMotionMap::evaluate(
    const std::array<real_t, 3>& reference_point,
    const std::array<real_t, 3>& previous_point,
    const MotionMapTimeState&) const {
  const auto r = sub(reference_point, parameters_.origin);
  const auto current = add(add(parameters_.origin, matvec(parameters_.transform, r)),
                           parameters_.translation);
  const auto velocity = add(parameters_.linear_velocity,
                            matvec(parameters_.velocity_gradient, r));
  const auto acceleration = add(parameters_.linear_acceleration,
                                matvec(parameters_.acceleration_gradient, r));

  MotionMapPointState state;
  state.reference_point = reference_point;
  state.previous_point = previous_point;
  state.current_point = current;
  state.displacement = sub(current, reference_point);
  state.velocity = velocity;
  state.acceleration = acceleration;
  return state;
}

std::vector<index_t> select_motion_map_geometry_dofs(
    const MeshBase& mesh,
    const MotionMapTarget& target) {
  std::vector<index_t> dofs;
  switch (target.kind) {
    case MotionMapTargetKind::AllGeometryDofs:
      dofs.reserve(mesh.geometry_dof_count());
      for (index_t i = 0; i < static_cast<index_t>(mesh.geometry_dof_count()); ++i) {
        dofs.push_back(i);
      }
      break;
    case MotionMapTargetKind::ExplicitGeometryDofs:
      dofs = target.geometry_dofs;
      break;
    case MotionMapTargetKind::BoundaryLabel:
      for (const index_t face : mesh.faces_with_label(target.label)) {
        append_face_geometry_dofs(mesh, face, dofs);
      }
      break;
    case MotionMapTargetKind::RegionLabel:
      for (const index_t cell : mesh.cells_with_label(target.label)) {
        append_cell_geometry_dofs(mesh, cell, dofs);
      }
      break;
    case MotionMapTargetKind::VertexLabel:
      dofs = mesh.vertices_with_label(target.label);
      break;
  }
  sort_unique_validate(mesh, dofs);
  return dofs;
}

MotionMapApplyResult apply_motion_map(MeshBase& mesh,
                                      const IMotionMap& motion_map,
                                      const MotionMapTarget& target,
                                      const MotionMapTimeState& time_state,
                                      const MotionMapApplyOptions& options) {
  if (mesh.dim() <= 0) {
    throw std::runtime_error("apply_motion_map: mesh dimension must be positive");
  }

  MotionMapApplyResult result;
  result.kind = motion_map.kind();
  result.target = target;
  result.time_state = time_state;
  result.geometry_revision_before = mesh.geometry_revision();
  result.active_configuration_epoch_before = mesh.active_configuration_epoch();
  result.geometry_dofs = select_motion_map_geometry_dofs(mesh, target);
  if (result.geometry_dofs.empty() && !options.allow_empty_target) {
    throw std::runtime_error("apply_motion_map: motion target selected no geometry DOFs");
  }

  result.dof_states.reserve(result.geometry_dofs.size());
  for (const index_t dof : result.geometry_dofs) {
    const auto reference_point = mesh.geometry_dof_coords(dof, Configuration::Reference);
    const auto previous_point = mesh.geometry_dof_coords(
        dof, mesh.has_current_coords() ? Configuration::Current : Configuration::Reference);
    auto state = motion_map.evaluate(reference_point, previous_point, time_state);
    if (options.require_finite_values &&
        (!finite_vec(state.current_point) ||
         !finite_vec(state.displacement) ||
         !finite_vec(state.velocity) ||
         !finite_vec(state.acceleration))) {
      throw std::runtime_error("apply_motion_map: motion map produced non-finite values");
    }
    result.dof_states.push_back(state);
  }

  if (options.update_current_coordinates) {
    std::vector<real_t> coords = mesh.has_current_coords() ? mesh.X_cur() : mesh.X_ref();
    const size_t dim = static_cast<size_t>(mesh.dim());
    if (coords.size() != mesh.X_ref().size()) {
      throw std::runtime_error("apply_motion_map: current/reference coordinate size mismatch");
    }
    for (size_t i = 0; i < result.geometry_dofs.size(); ++i) {
      const size_t base = static_cast<size_t>(result.geometry_dofs[i]) * dim;
      for (size_t d = 0; d < dim && d < 3u; ++d) {
        coords[base + d] = result.dof_states[i].current_point[d];
      }
    }
    mesh.set_current_coords(coords);
    if (options.set_active_configuration_current) {
      mesh.use_current_configuration();
    }
  }

  publish_motion_fields(mesh, result, options);

  result.geometry_revision_after = mesh.geometry_revision();
  result.active_configuration_epoch_after = mesh.active_configuration_epoch();
  return result;
}

MotionMapApplyResult apply_motion_map(Mesh& mesh,
                                      const IMotionMap& motion_map,
                                      const MotionMapTarget& target,
                                      const MotionMapTimeState& time_state,
                                      const MotionMapApplyOptions& options) {
  auto result = apply_motion_map(mesh.local_mesh(), motion_map, target, time_state, options);
  if (options.update_current_coordinates) {
    mesh.update_exchange_ghost_coordinates(Configuration::Current);
    result.geometry_revision_after = mesh.local_mesh().geometry_revision();
  }
  if (options.update_motion_fields) {
    mesh.update_exchange_ghost_fields();
  }
  return result;
}

MotionMapRestartRecord make_motion_map_restart_record(
    const MeshBase& mesh,
    const IMotionMap& motion_map,
    const MotionMapTarget& target,
    const MotionMapTimeState& time_state) {
  MotionMapRestartRecord record;
  record.map_name = motion_map.name();
  record.map_kind = motion_map.kind();
  record.target = target;
  record.time_state = time_state;
  record.geometry_revision = mesh.geometry_revision();
  record.topology_revision = mesh.topology_revision();
  record.ownership_revision = mesh.ownership_revision();
  record.numbering_revision = mesh.numbering_revision();
  record.field_layout_revision = mesh.field_layout_revision();
  record.label_revision = mesh.label_revision();
  record.active_configuration_epoch = mesh.active_configuration_epoch();
  return record;
}

MotionMapTransaction::MotionMapTransaction(MeshBase& mesh)
    : mesh_(&mesh)
    , entry_geometry_revision_(mesh.geometry_revision())
    , state_(MotionMapTransactionState::Trial) {
  save_coordinates(mesh, backup_);
}

MotionMapApplyResult MotionMapTransaction::apply(
    const IMotionMap& motion_map,
    const MotionMapTarget& target,
    const MotionMapTimeState& time_state,
    const MotionMapApplyOptions& options) {
  if (!mesh_) {
    throw std::runtime_error("MotionMapTransaction::apply: transaction has no mesh");
  }
  if (state_ == MotionMapTransactionState::Accepted) {
    throw std::runtime_error("MotionMapTransaction::apply: transaction is already accepted");
  }
  if (state_ == MotionMapTransactionState::RolledBack) {
    throw std::runtime_error("MotionMapTransaction::apply: transaction is already rolled back");
  }
  auto result = apply_motion_map(*mesh_, motion_map, target, time_state, options);
  result.transaction_state = MotionMapTransactionState::Trial;
  state_ = MotionMapTransactionState::Trial;
  return result;
}

void MotionMapTransaction::accept() {
  if (!mesh_) {
    return;
  }
  state_ = MotionMapTransactionState::Accepted;
}

void MotionMapTransaction::rollback() {
  if (!mesh_ || state_ == MotionMapTransactionState::RolledBack) {
    return;
  }
  restore_coordinates(*mesh_, backup_);
  state_ = MotionMapTransactionState::RolledBack;
}

} // namespace motion
} // namespace svmp
