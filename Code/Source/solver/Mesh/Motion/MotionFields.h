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

#ifndef SVMP_MOTION_FIELDS_H
#define SVMP_MOTION_FIELDS_H

/**
 * @file MotionFields.h
 * @brief Mesh-attached fields for mesh motion and moving-domain time history.
 *
 * This header declares lightweight helpers for attaching and updating
 * mesh-motion related fields on top of MeshBase. It operates purely on
 * geometry and Mesh's field system, without introducing FE/DOF concepts.
 */

#include "../Core/MeshTypes.h"
#include "../Fields/MeshFieldDescriptor.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace svmp {

class MeshBase;
class DistributedMesh;
// Phase 5 (UNIFY_MESH): prefer the unified runtime mesh type name.
// In the Mesh library, `Mesh` is currently an alias of `DistributedMesh`.
using Mesh = DistributedMesh;

namespace motion {

enum class MotionFieldRole : std::uint8_t {
  Displacement,
  Velocity,
  Acceleration,
  PreviousCoordinates,
  PreviousDisplacement,
  PreviousVelocity
};

enum class MotionFieldTimeLevel : std::uint8_t {
  Current,
  Previous
};

enum class DisplacementUpdateMode : std::uint8_t {
  AbsoluteFromReference,
  IncrementalFromCurrent
};

struct MotionFieldMetadata {
  MotionFieldRole role{MotionFieldRole::Displacement};
  MotionFieldTimeLevel time_level{MotionFieldTimeLevel::Current};
  std::string_view name{};
  std::string_view units{};
  std::string_view description{};
  FieldScalarType scalar_type{FieldScalarType::Float64};
  EntityKind location{EntityKind::Vertex};
  FieldGhostPolicy ghost_policy{FieldGhostPolicy::Exchange};
  bool time_dependent{true};
};

/**
 * @brief Handles to mesh-motion related fields attached to a mesh.
 *
 * All standard fields live on vertices, use Float64 storage, have one vector
 * component per spatial dimension, and use direct ghost exchange. The current
 * displacement is used as an accepted-step increment by MeshMotion; the
 * `update_coordinates_from_displacement` API below explicitly selects whether
 * a displacement field is interpreted as absolute-from-reference or
 * incremental-from-current for direct coordinate updates.
 */
struct MotionFieldHandles {
  FieldHandle displacement;
  FieldHandle velocity;
  FieldHandle acceleration;
  FieldHandle previous_coordinates;
  FieldHandle previous_displacement;
  FieldHandle previous_velocity;
};

[[nodiscard]] const char* to_string(MotionFieldRole role) noexcept;
[[nodiscard]] const char* to_string(DisplacementUpdateMode mode) noexcept;

[[nodiscard]] MotionFieldRole parse_motion_field_role(std::string_view role_name);
[[nodiscard]] bool is_standard_motion_field_name(std::string_view field_name) noexcept;
[[nodiscard]] MotionFieldMetadata standard_motion_field_metadata(MotionFieldRole role) noexcept;
[[nodiscard]] const char* standard_motion_field_name(MotionFieldRole role) noexcept;
[[nodiscard]] std::vector<MotionFieldRole> standard_motion_field_roles();

[[nodiscard]] FieldDescriptor standard_motion_field_descriptor(MotionFieldRole role, int nsd);

FieldHandle ensure_motion_field(MeshBase& mesh, MotionFieldRole role, int nsd);
FieldHandle ensure_motion_field(Mesh& mesh, MotionFieldRole role, int nsd);

void validate_motion_field(const MeshBase& mesh,
                           const FieldHandle& field,
                           MotionFieldRole role,
                           int nsd);
void validate_motion_field(const Mesh& mesh,
                           const FieldHandle& field,
                           MotionFieldRole role,
                           int nsd);

/**
 * @brief Attach (or lookup) standard mesh-motion fields on a mesh.
 *
 * This function ensures that displacement, velocity, acceleration, previous
 * coordinates, previous displacement, and previous velocity fields exist on
 * the given mesh as vertex-attached Float64 arrays with @p nsd components.
 * Existing fields are validated against the standard contract before their
 * handles are returned.
 *
 * @param mesh Mesh on which to attach motion fields.
 * @param nsd  Spatial dimension (typically 2 or 3).
 * @return Handles to the displacement and velocity fields.
 *
 * @throws std::runtime_error if @p nsd is incompatible with the mesh.
 */
MotionFieldHandles attach_motion_fields(MeshBase& mesh, int nsd);
MotionFieldHandles attach_motion_fields(Mesh& mesh, int nsd);

/**
 * @brief Update mesh coordinates from a displacement field.
 *
 * The displacement field is interpreted in physical space as one vector
 * per vertex. When @p accumulate is false,
 * the new coordinates are computed as
 *
 *   X_cur = X_ref + u
 *
 * using the reference configuration as the base. When @p accumulate
 * is true and a current coordinate field already exists, the update
 * becomes
 *
 *   X_cur_new = X_cur_old + u
 *
 * which is useful for incremental motion schemes; if no current
 * coordinates exist yet, the reference configuration is used as the
 * base instead.
 *
 * This function only updates geometry; it does not modify the
 * displacement or velocity fields themselves.
 *
 * @param mesh       Mesh whose coordinates will be updated.
 * @param hnd        Handles to motion fields (displacement is required).
 * @param accumulate If true, displacement is applied on top of the
 *                   current configuration when available.
 */
void update_coordinates_from_displacement(MeshBase& mesh,
                                          const MotionFieldHandles& hnd,
                                          DisplacementUpdateMode mode);
void update_coordinates_from_displacement(Mesh& mesh,
                                          const MotionFieldHandles& hnd,
                                          DisplacementUpdateMode mode);

void update_coordinates_from_displacement(MeshBase& mesh,
                                          const MotionFieldHandles& hnd,
                                          bool accumulate);
void update_coordinates_from_displacement(Mesh& mesh,
                                          const MotionFieldHandles& hnd,
                                          bool accumulate);

} // namespace motion
} // namespace svmp

#endif // SVMP_MOTION_FIELDS_H
