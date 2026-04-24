/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "MotionFields.h"

#include "../Core/MeshBase.h"
#include "../Core/DistributedMesh.h"
#include "../Fields/MeshFields.h"
#include "../Fields/MeshFieldDescriptor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace svmp {
namespace motion {

namespace {

constexpr std::array<MotionFieldRole, 6> kStandardMotionRoles = {{
    MotionFieldRole::Displacement,
    MotionFieldRole::Velocity,
    MotionFieldRole::Acceleration,
    MotionFieldRole::PreviousCoordinates,
    MotionFieldRole::PreviousDisplacement,
    MotionFieldRole::PreviousVelocity,
}};

void validate_nsd(const MeshBase& mesh, int nsd, const char* caller)
{
  if (nsd <= 0) {
    throw std::invalid_argument(std::string(caller) + ": nsd must be positive");
  }

  int mesh_dim = mesh.dim();
  if (mesh_dim > 0 && mesh_dim != nsd) {
    throw std::runtime_error(std::string(caller) + ": nsd does not match mesh dimension");
  }
}

void check_finite_field(const real_t* data,
                        size_t n_entities,
                        size_t components,
                        int dim,
                        const char* caller)
{
  if (!data) {
    throw std::runtime_error(std::string(caller) + ": field data is null");
  }
  for (size_t v = 0; v < n_entities; ++v) {
    const size_t base = v * components;
    for (int k = 0; k < dim; ++k) {
      if (!std::isfinite(static_cast<double>(data[base + static_cast<size_t>(k)]))) {
        throw std::runtime_error(std::string(caller) + ": field contains a non-finite value");
      }
    }
  }
}

} // namespace

const char* to_string(MotionFieldRole role) noexcept
{
  switch (role) {
    case MotionFieldRole::Displacement: return "mesh_displacement";
    case MotionFieldRole::Velocity: return "mesh_velocity";
    case MotionFieldRole::Acceleration: return "mesh_acceleration";
    case MotionFieldRole::PreviousCoordinates: return "previous_coordinates";
    case MotionFieldRole::PreviousDisplacement: return "previous_mesh_displacement";
    case MotionFieldRole::PreviousVelocity: return "previous_mesh_velocity";
  }
  return "unknown_motion_field";
}

const char* to_string(DisplacementUpdateMode mode) noexcept
{
  switch (mode) {
    case DisplacementUpdateMode::AbsoluteFromReference:
      return "absolute_from_reference";
    case DisplacementUpdateMode::IncrementalFromCurrent:
      return "incremental_from_current";
  }
  return "unknown_displacement_update_mode";
}

MotionFieldRole parse_motion_field_role(std::string_view role_name)
{
  if (role_name == "mesh_displacement" || role_name == "displacement") {
    return MotionFieldRole::Displacement;
  }
  if (role_name == "mesh_velocity" || role_name == "velocity") {
    return MotionFieldRole::Velocity;
  }
  if (role_name == "mesh_acceleration" || role_name == "acceleration") {
    return MotionFieldRole::Acceleration;
  }
  if (role_name == "previous_coordinates" ||
      role_name == "previous_coordinate" ||
      role_name == "previous_coords") {
    return MotionFieldRole::PreviousCoordinates;
  }
  if (role_name == "previous_mesh_displacement" ||
      role_name == "mesh_displacement_previous" ||
      role_name == "previous_displacement") {
    return MotionFieldRole::PreviousDisplacement;
  }
  if (role_name == "previous_mesh_velocity" ||
      role_name == "mesh_velocity_previous" ||
      role_name == "previous_velocity") {
    return MotionFieldRole::PreviousVelocity;
  }
  throw std::invalid_argument("parse_motion_field_role: unknown motion field role '" +
                              std::string(role_name) + "'");
}

bool is_standard_motion_field_name(std::string_view field_name) noexcept
{
  for (auto role : kStandardMotionRoles) {
    if (field_name == standard_motion_field_name(role)) {
      return true;
    }
  }
  return false;
}

MotionFieldMetadata standard_motion_field_metadata(MotionFieldRole role) noexcept
{
  switch (role) {
    case MotionFieldRole::Displacement:
      return MotionFieldMetadata{
          role,
          MotionFieldTimeLevel::Current,
          "mesh_displacement",
          "m",
          "Mesh displacement vector; MeshMotion publishes accepted-step increments, while direct coordinate updates select absolute or incremental interpretation explicitly.",
          FieldScalarType::Float64,
          EntityKind::Vertex,
          FieldGhostPolicy::Exchange,
          true};
    case MotionFieldRole::Velocity:
      return MotionFieldMetadata{
          role,
          MotionFieldTimeLevel::Current,
          "mesh_velocity",
          "m/s",
          "Mesh velocity vector at the current accepted mesh state.",
          FieldScalarType::Float64,
          EntityKind::Vertex,
          FieldGhostPolicy::Exchange,
          true};
    case MotionFieldRole::Acceleration:
      return MotionFieldMetadata{
          role,
          MotionFieldTimeLevel::Current,
          "mesh_acceleration",
          "m/s^2",
          "Mesh acceleration vector at the current accepted mesh state.",
          FieldScalarType::Float64,
          EntityKind::Vertex,
          FieldGhostPolicy::Exchange,
          true};
    case MotionFieldRole::PreviousCoordinates:
      return MotionFieldMetadata{
          role,
          MotionFieldTimeLevel::Previous,
          "previous_coordinates",
          "m",
          "Mesh coordinates at the previous accepted moving-mesh state.",
          FieldScalarType::Float64,
          EntityKind::Vertex,
          FieldGhostPolicy::Exchange,
          true};
    case MotionFieldRole::PreviousDisplacement:
      return MotionFieldMetadata{
          role,
          MotionFieldTimeLevel::Previous,
          "previous_mesh_displacement",
          "m",
          "Mesh displacement vector from the previous accepted moving-mesh state.",
          FieldScalarType::Float64,
          EntityKind::Vertex,
          FieldGhostPolicy::Exchange,
          true};
    case MotionFieldRole::PreviousVelocity:
      return MotionFieldMetadata{
          role,
          MotionFieldTimeLevel::Previous,
          "previous_mesh_velocity",
          "m/s",
          "Mesh velocity vector from the previous accepted moving-mesh state.",
          FieldScalarType::Float64,
          EntityKind::Vertex,
          FieldGhostPolicy::Exchange,
          true};
  }
  return standard_motion_field_metadata(MotionFieldRole::Displacement);
}

const char* standard_motion_field_name(MotionFieldRole role) noexcept
{
  return to_string(role);
}

std::vector<MotionFieldRole> standard_motion_field_roles()
{
  return std::vector<MotionFieldRole>(kStandardMotionRoles.begin(), kStandardMotionRoles.end());
}

FieldDescriptor standard_motion_field_descriptor(MotionFieldRole role, int nsd)
{
  if (nsd <= 0) {
    throw std::invalid_argument("standard_motion_field_descriptor: nsd must be positive");
  }
  const auto metadata = standard_motion_field_metadata(role);
  FieldDescriptor desc = FieldDescriptor::vector(metadata.location,
                                                 nsd,
                                                 std::string(metadata.units),
                                                 metadata.time_dependent);
  desc.intent = FieldIntent::ReadWrite;
  desc.ghost_policy = metadata.ghost_policy;
  desc.description = std::string(metadata.description);
  return desc;
}

FieldHandle ensure_motion_field(MeshBase& mesh, MotionFieldRole role, int nsd)
{
  validate_nsd(mesh, nsd, "ensure_motion_field");
  const auto metadata = standard_motion_field_metadata(role);
  const std::string name(metadata.name);

  FieldHandle h;
  if (MeshFields::has_field(mesh, metadata.location, name)) {
    h = MeshFields::get_field_handle(mesh, metadata.location, name);
    const auto* desc = MeshFields::field_descriptor(mesh, h);
    if (!desc) {
      if (MeshFields::field_type(mesh, h) != metadata.scalar_type ||
          MeshFields::field_components(mesh, h) != static_cast<size_t>(nsd) ||
          MeshFields::field_entity_count(mesh, h) != mesh.n_vertices()) {
        throw std::runtime_error("ensure_motion_field: existing field '" + name +
                                 "' is incompatible with the standard motion-field contract");
      }
      mesh.set_field_descriptor(h, standard_motion_field_descriptor(role, nsd));
    }
  } else {
    h = MeshFields::attach_field_with_descriptor(
        mesh,
        metadata.location,
        name,
        metadata.scalar_type,
        standard_motion_field_descriptor(role, nsd));
  }
  validate_motion_field(mesh, h, role, nsd);
  return h;
}

FieldHandle ensure_motion_field(Mesh& mesh, MotionFieldRole role, int nsd)
{
  return ensure_motion_field(mesh.local_mesh(), role, nsd);
}

void validate_motion_field(const MeshBase& mesh,
                           const FieldHandle& field,
                           MotionFieldRole role,
                           int nsd)
{
  validate_nsd(mesh, nsd, "validate_motion_field");
  const auto metadata = standard_motion_field_metadata(role);

  if (field.id == 0) {
    throw std::runtime_error("validate_motion_field: invalid field handle for " +
                             std::string(metadata.name));
  }
  if (field.kind != metadata.location) {
    throw std::runtime_error("validate_motion_field: field '" +
                             std::string(metadata.name) +
                             "' has the wrong entity location");
  }
  if (MeshFields::field_type(mesh, field) != metadata.scalar_type) {
    throw std::runtime_error("validate_motion_field: field '" +
                             std::string(metadata.name) +
                             "' must use Float64 storage");
  }
  if (MeshFields::field_components(mesh, field) != static_cast<size_t>(nsd)) {
    throw std::runtime_error("validate_motion_field: field '" +
                             std::string(metadata.name) +
                             "' component count must match mesh dimension");
  }
  if (MeshFields::field_entity_count(mesh, field) != mesh.n_vertices()) {
    throw std::runtime_error("validate_motion_field: field '" +
                             std::string(metadata.name) +
                             "' entity count does not match mesh vertices");
  }

  const auto* desc = MeshFields::field_descriptor(mesh, field);
  if (!desc) {
    throw std::runtime_error("validate_motion_field: field '" +
                             std::string(metadata.name) +
                             "' is missing standard metadata");
  }
  if (desc->location != metadata.location ||
      desc->components != static_cast<size_t>(nsd) ||
      desc->units != std::string(metadata.units) ||
      desc->time_dependent != metadata.time_dependent ||
      desc->ghost_policy != metadata.ghost_policy) {
    throw std::runtime_error("validate_motion_field: field '" +
                             std::string(metadata.name) +
                             "' metadata does not match the standard motion-field contract");
  }
}

void validate_motion_field(const Mesh& mesh,
                           const FieldHandle& field,
                           MotionFieldRole role,
                           int nsd)
{
  validate_motion_field(mesh.local_mesh(), field, role, nsd);
}

MotionFieldHandles attach_motion_fields(MeshBase& mesh, int nsd)
{
  validate_nsd(mesh, nsd, "attach_motion_fields");

  MotionFieldHandles hnd;
  hnd.displacement = ensure_motion_field(mesh, MotionFieldRole::Displacement, nsd);
  hnd.velocity = ensure_motion_field(mesh, MotionFieldRole::Velocity, nsd);
  hnd.acceleration = ensure_motion_field(mesh, MotionFieldRole::Acceleration, nsd);
  hnd.previous_coordinates = ensure_motion_field(mesh, MotionFieldRole::PreviousCoordinates, nsd);
  hnd.previous_displacement = ensure_motion_field(mesh, MotionFieldRole::PreviousDisplacement, nsd);
  hnd.previous_velocity = ensure_motion_field(mesh, MotionFieldRole::PreviousVelocity, nsd);

  return hnd;
}

void update_coordinates_from_displacement(MeshBase& mesh,
                                          const MotionFieldHandles& hnd,
                                          DisplacementUpdateMode mode)
{
  if (hnd.displacement.id == 0) {
    throw std::runtime_error(
        "update_coordinates_from_displacement: displacement field handle is invalid");
  }

  const int dim = mesh.dim();
  if (dim <= 0) {
    // Degenerate or uninitialized meshes are not supported here.
    throw std::runtime_error(
        "update_coordinates_from_displacement: mesh dimension must be positive");
  }

  validate_motion_field(mesh, hnd.displacement, MotionFieldRole::Displacement, dim);

  const size_t n_vertices = mesh.n_vertices();
  if (n_vertices == 0) {
    return; // Nothing to do
  }

  const size_t disp_components = MeshFields::field_components(mesh, hnd.displacement);
  if (disp_components < static_cast<size_t>(dim)) {
    throw std::runtime_error(
        "update_coordinates_from_displacement: displacement field has fewer components "
        "than the mesh spatial dimension");
  }

  const real_t* disp = MeshFields::field_data_as<real_t>(mesh, hnd.displacement);
  check_finite_field(disp, n_vertices, disp_components, dim,
                     "update_coordinates_from_displacement");

  const std::vector<real_t>& X_ref = mesh.X_ref();
  if (X_ref.empty()) {
    throw std::runtime_error(
        "update_coordinates_from_displacement: mesh has no reference coordinates");
  }
  if (X_ref.size() != n_vertices * static_cast<size_t>(dim)) {
    throw std::runtime_error(
        "update_coordinates_from_displacement: reference coordinate array size mismatch");
  }

  // Choose the base configuration for the update.
  const bool use_current_as_base =
      (mode == DisplacementUpdateMode::IncrementalFromCurrent) && mesh.has_current_coords();
  const std::vector<real_t>& X_base = (use_current_as_base && !mesh.X_cur().empty())
      ? mesh.X_cur()
      : X_ref;

  if (X_base.size() != X_ref.size()) {
    throw std::runtime_error(
        "update_coordinates_from_displacement: coordinate array size mismatch");
  }

  std::vector<real_t> X_new(X_ref.size());

  for (size_t v = 0; v < n_vertices; ++v) {
    const size_t coord_offset = v * static_cast<size_t>(dim);
    const size_t disp_offset  = v * disp_components;

    for (int k = 0; k < dim; ++k) {
      const real_t base = X_base[coord_offset + static_cast<size_t>(k)];
      const real_t du   = disp[disp_offset + static_cast<size_t>(k)];
      X_new[coord_offset + static_cast<size_t>(k)] = base + du;
    }
  }

  mesh.set_current_coords(X_new);
  mesh.use_current_configuration();
}

MotionFieldHandles attach_motion_fields(Mesh& mesh, int nsd)
{
  return attach_motion_fields(mesh.local_mesh(), nsd);
}

void update_coordinates_from_displacement(Mesh& mesh,
                                          const MotionFieldHandles& hnd,
                                          DisplacementUpdateMode mode)
{
  update_coordinates_from_displacement(mesh.local_mesh(), hnd, mode);
}

void update_coordinates_from_displacement(MeshBase& mesh,
                                          const MotionFieldHandles& hnd,
                                          bool accumulate)
{
  update_coordinates_from_displacement(
      mesh,
      hnd,
      accumulate ? DisplacementUpdateMode::IncrementalFromCurrent
                 : DisplacementUpdateMode::AbsoluteFromReference);
}

void update_coordinates_from_displacement(Mesh& mesh,
                                          const MotionFieldHandles& hnd,
                                          bool accumulate)
{
  update_coordinates_from_displacement(mesh.local_mesh(), hnd, accumulate);
}

} // namespace motion
} // namespace svmp
