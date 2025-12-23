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

#ifndef SVMP_MESH_MOTION_H
#define SVMP_MESH_MOTION_H

/**
 * @file MeshMotion.h
 * @brief High-level controller for mesh motion on MeshBase / DistributedMesh.
 *
 * MeshMotion provides a backend-agnostic interface for advancing mesh
 * coordinates in time. Motion models, weak forms, and solver choices are
 * provided by injected backends (e.g., future Physics libraries). MeshMotion
 * itself owns Mesh-side responsibilities: motion fields, coordinate updates,
 * step acceptance/rollback, quality guards, and ghost synchronization.
 */

#include "../Core/MeshTypes.h"
#include "IMotionBackend.h"
#include "MotionConfig.h"

#include <memory>
#include <vector>

namespace svmp {

class MeshBase;
class DistributedMesh;

namespace motion {

/**
 * @brief High-level mesh-motion controller.
 *
 * This class wraps either a MeshBase (serial) or a DistributedMesh
 * (parallel) instance and provides a uniform interface to advance
 * the mesh coordinates via configured motion models.
 *
 * MeshMotion is intentionally FE-agnostic: it owns the mesh-side geometry
 * and field plumbing and delegates the computation of motion fields
 * (displacement and optionally velocity) to an injected IMotionBackend.
 * The backend may live in another library (e.g., FE) and is supplied via
 * explicit dependency injection.
 *
 * Motion quality guards, backtracking, and sub-stepping are handled inside
 * advance() when enabled via MotionConfig.
 */
class MeshMotion {
public:
  /// Construct a MeshMotion wrapper around a serial mesh.
  explicit MeshMotion(MeshBase& mesh);

  /// Construct a MeshMotion wrapper around a distributed mesh.
  explicit MeshMotion(DistributedMesh& dmesh);

  /// Set the configuration for mesh motion.
  void set_config(const MotionConfig& cfg) { cfg_ = cfg; }

  /// Access the current configuration.
  const MotionConfig& config() const noexcept { return cfg_; }

  /// Inject (or clear) the backend used to compute motion fields.
  void set_backend(std::shared_ptr<IMotionBackend> backend) { backend_ = std::move(backend); }
  [[nodiscard]] bool has_backend() const noexcept { return static_cast<bool>(backend_); }
  [[nodiscard]] IMotionBackend* backend() noexcept { return backend_.get(); }
  [[nodiscard]] const IMotionBackend* backend() const noexcept { return backend_.get(); }

  /// Set (or clear) Dirichlet boundary conditions for mesh motion.
  void set_dirichlet_bcs(std::vector<MotionDirichletBC> bcs) { dirichlet_bcs_ = std::move(bcs); }
  void clear_dirichlet_bcs() { dirichlet_bcs_.clear(); }
  [[nodiscard]] const std::vector<MotionDirichletBC>& dirichlet_bcs() const noexcept { return dirichlet_bcs_; }

  /// Access the underlying mesh as a MeshBase view (local mesh for
  /// distributed meshes). This always returns a valid reference.
  MeshBase& mesh();
  const MeshBase& mesh() const;

  /// Return true if this controller wraps a DistributedMesh.
  bool has_distributed_mesh() const noexcept { return dmesh_ != nullptr; }

  /// Access the underlying DistributedMesh, if any (may be null).
  DistributedMesh*       distributed_mesh()       noexcept { return dmesh_; }
  const DistributedMesh* distributed_mesh() const noexcept { return dmesh_; }

  /**
   * @brief Advance the mesh coordinates by a time increment dt.
   *
   * This method orchestrates a single motion step:
   *  - Ensures a current-coordinate field exists (initializes to X_ref if needed).
   *  - Ensures the standard motion fields exist on vertices.
   *  - If a backend is injected: calls backend->solve() to compute displacement
   *    increments over one or more substeps (step_scale fractions of dt), applies
   *    them on top of X_cur, and accumulates the total increment for the call.
   *  - If the backend reports failure, restores the coordinate state to what it
   *    was at entry and returns false.
   *  - If quality guards are enabled, rejected substeps are rolled back and
   *    retried with smaller step_scale values up to config limits.
   *  - If the backend does not provide velocity (or if multiple substeps are
   *    used) and dt > 0, computes velocity = displacement / dt.
   *
   * If no backend is injected, this function is a no-op and returns true.
   *
   * @param dt Time increment (passed to backend; used for velocity fallback).
   * @return true if the step was accepted, false if rejected.
   */
  bool advance(double dt);

  /**
   * @brief Reset the mesh to its reference configuration.
   *
   * This clears any current-coordinate field and marks the active
   * configuration as reference.
   */
  void reset_to_reference();

private:
  // Non-owning pointers to the underlying mesh objects. Exactly one of
  // mesh_ or dmesh_ is non-null, depending on how this controller was
  // constructed.
  MeshBase*        mesh_  = nullptr;
  DistributedMesh* dmesh_ = nullptr;

  MotionConfig cfg_;
  std::shared_ptr<IMotionBackend> backend_;
  std::vector<MotionDirichletBC> dirichlet_bcs_;
};

} // namespace motion
} // namespace svmp

#endif // SVMP_MESH_MOTION_H
