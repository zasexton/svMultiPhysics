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

#include <stdexcept>
#include <vector>

namespace svmp {
namespace motion {

MotionFieldHandles attach_motion_fields(MeshBase& mesh, int nsd)
{
  if (nsd <= 0) {
    throw std::invalid_argument("attach_motion_fields: nsd must be positive");
  }

  // If mesh has a known spatial dimension, ensure nsd is compatible.
  int mesh_dim = mesh.dim();
  if (mesh_dim > 0 && mesh_dim != nsd) {
    throw std::runtime_error("attach_motion_fields: nsd does not match mesh dimension");
  }

  MotionFieldHandles hnd;

  // Displacement field u (used as a step increment by MeshMotion)
  const std::string disp_name = "mesh_displacement";
  if (MeshFields::has_field(mesh, EntityKind::Vertex, disp_name)) {
    hnd.displacement = MeshFields::get_field_handle(mesh, EntityKind::Vertex, disp_name);
  } else {
    FieldDescriptor desc = FieldDescriptor::vector(EntityKind::Vertex, nsd, "m", true);
    desc.intent = FieldIntent::ReadWrite;
    desc.ghost_policy = FieldGhostPolicy::Exchange;
    desc.description = "Mesh displacement increment u attached to vertices";

    hnd.displacement = MeshFields::attach_field_with_descriptor(
        mesh,
        EntityKind::Vertex,
        disp_name,
        FieldScalarType::Float64,
        desc);
  }

  // Velocity field w(X,t)
  const std::string vel_name = "mesh_velocity";
  if (MeshFields::has_field(mesh, EntityKind::Vertex, vel_name)) {
    hnd.velocity = MeshFields::get_field_handle(mesh, EntityKind::Vertex, vel_name);
  } else {
    FieldDescriptor desc = FieldDescriptor::vector(EntityKind::Vertex, nsd, "m/s", true);
    desc.intent = FieldIntent::ReadWrite;
    desc.ghost_policy = FieldGhostPolicy::Exchange;
    desc.description = "Mesh velocity w(X,t) attached to vertices";

    hnd.velocity = MeshFields::attach_field_with_descriptor(
        mesh,
        EntityKind::Vertex,
        vel_name,
        FieldScalarType::Float64,
        desc);
  }

  return hnd;
}

void update_coordinates_from_displacement(MeshBase& mesh,
                                          const MotionFieldHandles& hnd,
                                          bool accumulate)
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
  if (!disp) {
    throw std::runtime_error(
        "update_coordinates_from_displacement: displacement field data is null");
  }

  const std::vector<real_t>& X_ref = mesh.X_ref();
  if (X_ref.empty()) {
    throw std::runtime_error(
        "update_coordinates_from_displacement: mesh has no reference coordinates");
  }

  // Choose the base configuration for the update.
  const bool use_current_as_base = accumulate && mesh.has_current_coords();
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
                                          bool accumulate)
{
  update_coordinates_from_displacement(mesh.local_mesh(), hnd, accumulate);
}

} // namespace motion
} // namespace svmp
