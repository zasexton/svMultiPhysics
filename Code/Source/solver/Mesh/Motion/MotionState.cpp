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

#include "MotionState.h"

#include "../Core/MeshBase.h"
#include "../Core/DistributedMesh.h"
#include "../Fields/MeshFields.h"

#include <stdexcept>

namespace svmp {
namespace motion {

void save_coordinates(const MeshBase& mesh, MotionCoordinateBackup& backup)
{
  backup.has_current   = mesh.has_current_coords();
  backup.active_config = mesh.active_configuration();
  backup.X_cur         = mesh.X_cur();
  backup.initialized   = true;
}

void save_coordinates(const Mesh& mesh, MotionCoordinateBackup& backup)
{
  save_coordinates(mesh.local_mesh(), backup);
}

void restore_coordinates(MeshBase& mesh, const MotionCoordinateBackup& backup)
{
  if (!backup.valid()) {
    // Nothing to restore; treat as no-op.
    return;
  }

  if (!backup.has_current) {
    mesh.clear_current_coords();
    mesh.use_reference_configuration();
    return;
  }

  mesh.set_current_coords(backup.X_cur);

  if (backup.active_config == Configuration::Reference) {
    mesh.use_reference_configuration();
  } else {
    // Treat Current/Deformed as the same active configuration.
    mesh.use_current_configuration();
  }
}

void restore_coordinates(Mesh& mesh, const MotionCoordinateBackup& backup)
{
  restore_coordinates(mesh.local_mesh(), backup);
}

void update_velocity_from_displacement(MeshBase& mesh,
                                       const MotionFieldHandles& hnd,
                                       real_t dt)
{
  if (dt <= 0.0) {
    throw std::invalid_argument("update_velocity_from_displacement: dt must be positive");
  }

  if (hnd.displacement.id == 0 || hnd.velocity.id == 0) {
    throw std::runtime_error(
        "update_velocity_from_displacement: displacement and velocity field handles must be valid");
  }

  const size_t n_vertices = mesh.n_vertices();
  if (n_vertices == 0) {
    return;
  }

  const size_t disp_components = MeshFields::field_components(mesh, hnd.displacement);
  const size_t vel_components  = MeshFields::field_components(mesh, hnd.velocity);

  if (disp_components == 0 || vel_components == 0) {
    throw std::runtime_error(
        "update_velocity_from_displacement: fields must have at least one component");
  }

  const size_t ncomp = (disp_components < vel_components) ? disp_components : vel_components;

  real_t* disp = MeshFields::field_data_as<real_t>(mesh, hnd.displacement);
  real_t* vel  = MeshFields::field_data_as<real_t>(mesh, hnd.velocity);

  if (!disp || !vel) {
    throw std::runtime_error(
        "update_velocity_from_displacement: field data pointers are null");
  }

  const real_t inv_dt = 1.0 / dt;

  for (size_t v = 0; v < n_vertices; ++v) {
    const size_t disp_offset = v * disp_components;
    const size_t vel_offset  = v * vel_components;

    for (size_t k = 0; k < ncomp; ++k) {
      vel[vel_offset + k] = disp[disp_offset + k] * inv_dt;
    }
  }
}

void update_velocity_from_displacement(Mesh& mesh,
                                       const MotionFieldHandles& hnd,
                                       real_t dt)
{
  update_velocity_from_displacement(mesh.local_mesh(), hnd, dt);
}

} // namespace motion
} // namespace svmp
