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

#include "MeshMotion.h"

#include "../Core/MeshBase.h"
#include "../Core/DistributedMesh.h"
#include "../Fields/MeshFields.h"

#include "MotionFields.h"
#include "MotionQuality.h"
#include "MotionState.h"

#include <algorithm>
#include <stdexcept>

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace motion {

namespace {

bool all_ranks_success(const Mesh& mesh, bool local_success)
{
#ifdef MESH_HAS_MPI
  if (mesh.world_size() <= 1) {
    return local_success;
  }

  const int local = local_success ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, mesh.mpi_comm());
  return global != 0;
#else
  (void)mesh;
  return local_success;
#endif
}

void add_displacement_to_current_coords(MeshBase& mesh,
                                        const real_t* displacement,
                                        size_t disp_components,
                                        int dim,
                                        real_t scale)
{
  if (!mesh.has_current_coords()) {
    throw std::runtime_error("add_displacement_to_current_coords: mesh has no current coordinates");
  }
  if (dim <= 0) {
    throw std::runtime_error("add_displacement_to_current_coords: mesh dimension must be positive");
  }
  if (!displacement) {
    throw std::runtime_error("add_displacement_to_current_coords: displacement pointer is null");
  }
  if (disp_components < static_cast<size_t>(dim)) {
    throw std::runtime_error("add_displacement_to_current_coords: displacement has too few components");
  }

  const size_t n_vertices = mesh.n_vertices();
  if (n_vertices == 0) {
    return;
  }

  auto* X_cur = mesh.X_cur_data_mutable();
  if (!X_cur) {
    throw std::runtime_error("add_displacement_to_current_coords: current coordinates buffer is null");
  }

  const size_t coord_stride = static_cast<size_t>(dim);
  for (size_t v = 0; v < n_vertices; ++v) {
    const size_t coord_offset = v * coord_stride;
    const size_t disp_offset  = v * disp_components;
    for (int k = 0; k < dim; ++k) {
      X_cur[coord_offset + static_cast<size_t>(k)] +=
          scale * displacement[disp_offset + static_cast<size_t>(k)];
    }
  }

  mesh.event_bus().notify(MeshEvent::GeometryChanged);
  mesh.use_current_configuration();
}

} // namespace

MeshMotion::MeshMotion(Mesh& mesh)
  : mesh_(&mesh)
{
}

MeshBase& MeshMotion::mesh()
{
  return mesh_->local_mesh();
}

const MeshBase& MeshMotion::mesh() const
{
  return mesh_->local_mesh();
}

bool MeshMotion::advance(double dt)
{
  MeshBase& mb = mesh();

  const bool entry_has_current = mb.has_current_coords();
  const Configuration entry_active_cfg = mb.active_configuration();

  // Ensure that the mesh has a current-coordinate field so incremental
  // updates can always be applied on top of X_cur().
  if (!mb.has_current_coords()) {
    mb.set_current_coords(mb.X_ref());
    mb.use_current_configuration();
  }

  // In distributed meshes, synchronize ghost vertex coordinates before the
  // solve so substep increments are applied on a consistent base geometry.
  if (mesh_->world_size() > 1) {
    mesh_->update_exchange_ghost_coordinates(Configuration::Current);
  }

  // Attach / lookup motion fields.
  const int dim = mb.dim();
  const auto hnd = attach_motion_fields(mb, dim);

  auto* disp = MeshFields::field_data_as<real_t>(mb, hnd.displacement);
  auto* vel  = MeshFields::field_data_as<real_t>(mb, hnd.velocity);

  MotionFieldView disp_view;
  MotionFieldView vel_view;

  disp_view.data = disp;
  disp_view.n_entities = mb.n_vertices();
  disp_view.components = MeshFields::field_components(mb, hnd.displacement);

  vel_view.data = vel;
  vel_view.n_entities = mb.n_vertices();
  vel_view.components = MeshFields::field_components(mb, hnd.velocity);

  const size_t n_vertices = mb.n_vertices();
  const size_t disp_size = n_vertices * disp_view.components;
  const size_t vel_size  = n_vertices * vel_view.components;

  const auto zero_motion_fields = [&]() {
    if (disp) std::fill(disp, disp + disp_size, real_t(0));
    if (vel) std::fill(vel, vel + vel_size, real_t(0));
  };

  // No backend: accept step, leaving coordinates unchanged.
  if (!backend_) {
    return true;
  }

  if (n_vertices == 0) {
    return true;
  }

  if (!disp_view.valid()) {
    return false;
  }

  // Track the total displacement over the full advance(dt) call (sum of accepted substeps).
  std::vector<real_t> total_disp(disp_size, real_t(0));

  // Substepping variables (step_scale is interpreted as a fraction of dt).
  double max_scale = cfg_.max_step_scale;
  if (!(max_scale > 0.0)) {
    max_scale = 1.0;
  }
  if (max_scale > 1.0) {
    max_scale = 1.0;
  }
  double current_max_scale = max_scale;
  double remaining_scale = 1.0;

  const auto restore_entry_state = [&]() {
    // Restore coordinates to the entry state by undoing the accumulated displacement.
    if (!entry_has_current) {
      mb.clear_current_coords();
      mb.use_reference_configuration();
    } else {
      if (!total_disp.empty()) {
        add_displacement_to_current_coords(mb, total_disp.data(), disp_view.components, dim, real_t(-1));
      }

      if (entry_active_cfg == Configuration::Reference) {
        mb.use_reference_configuration();
      } else {
        mb.use_current_configuration();
      }
    }
    zero_motion_fields();
  };

  std::vector<FieldHandle> exchange_disp;
  std::vector<FieldHandle> exchange_motion;
  if (mesh_->world_size() > 1) {
    exchange_disp = {hnd.displacement};
    exchange_motion = {hnd.displacement, hnd.velocity};
  }

  const int max_substeps = (cfg_.max_substeps > 0) ? cfg_.max_substeps : 1;
  const double eps_scale = 1e-12;

  int backend_calls = 0;
  int accepted_substeps = 0;
  bool last_wrote_velocity = false;

  while (remaining_scale > eps_scale) {
    if (backend_calls >= max_substeps) {
      restore_entry_state();
      return false;
    }

    const double step_scale = std::min(current_max_scale, remaining_scale);
    if (!(step_scale > eps_scale)) {
      restore_entry_state();
      return false;
    }

    // Ensure deterministic behavior even if the backend only writes partial fields.
    zero_motion_fields();

    const MotionSolveRequest req{
        *mesh_,
        cfg_,
        dt,
        step_scale,
        Configuration::Current,
        disp_view,
        vel_view,
        &dirichlet_bcs_};

    const MotionSolveResult result = backend_->solve(req);
    ++backend_calls;

    if (!all_ranks_success(*mesh_, result.success)) {
      restore_entry_state();
      return false;
    }

    last_wrote_velocity = result.wrote_velocity;

    // In distributed meshes, synchronize the displacement increment so owned
    // values propagate to ghost/shared vertices before coordinates are updated.
    if (mesh_->world_size() > 1 && !exchange_disp.empty()) {
      mesh_->update_ghosts(exchange_disp);
    }

    // Apply displacement increment to current coordinates.
    add_displacement_to_current_coords(mb, disp, disp_view.components, dim, real_t(1));

    // Quality gating / backtracking.
    if (cfg_.enable_quality_guard) {
      const MotionQualityReport report =
          evaluate_motion_quality(*mesh_, Configuration::Current);
      const real_t suggested = suggested_step_scale(
          report,
          static_cast<real_t>(step_scale),
          static_cast<real_t>(cfg_.quality_min_jacobian),
          static_cast<real_t>(cfg_.quality_min_angle_deg),
          static_cast<real_t>(cfg_.quality_max_skewness));

      const bool inverted_or_degenerate = (suggested <= real_t(0));
      const bool threshold_violation =
          cfg_.enforce_quality_thresholds && (suggested < static_cast<real_t>(step_scale));

      if (inverted_or_degenerate || threshold_violation) {
        // Undo this attempted increment and retry with a smaller substep.
        add_displacement_to_current_coords(mb, disp, disp_view.components, dim, real_t(-1));

        current_max_scale = inverted_or_degenerate
            ? (step_scale * 0.5)
            : static_cast<double>(suggested);
        continue;
      }

      // Optionally reduce subsequent substep sizes if quality is trending poor.
      current_max_scale = std::min(current_max_scale, static_cast<double>(suggested));
    }

    // Accepted: accumulate displacement increment into total.
    for (size_t i = 0; i < disp_size; ++i) {
      total_disp[i] += disp[i];
    }
    ++accepted_substeps;

    remaining_scale -= step_scale;
  }

  // Publish total step displacement.
  std::copy(total_disp.begin(), total_disp.end(), disp);

  // Publish mesh velocity.
  if (dt > 0.0) {
    // If we took multiple substeps, publish the average velocity over dt.
    // If we took a single step and the backend wrote velocity, preserve it.
    if (!(accepted_substeps == 1 && last_wrote_velocity)) {
      update_velocity_from_displacement(mb, hnd, static_cast<real_t>(dt));
    }
  } else {
    // dt <= 0: enforce deterministic output if no meaningful velocity can be derived.
    if (!(accepted_substeps == 1 && last_wrote_velocity)) {
      if (vel) {
        std::fill(vel, vel + vel_size, real_t(0));
      }
    }
  }

  // In distributed meshes, ensure ghost fields are synchronized according to
  // their ghost policies and exchange ghost vertex coordinates.
  if (mesh_->world_size() > 1) {
    if (!exchange_motion.empty()) {
      mesh_->update_ghosts(exchange_motion);
    }
    mesh_->update_exchange_ghost_coordinates(Configuration::Current);
  }

  return true;
}

void MeshMotion::reset_to_reference()
{
  MeshBase& mb = mesh();
  mb.clear_current_coords();
  mb.use_reference_configuration();
}

} // namespace motion
} // namespace svmp
