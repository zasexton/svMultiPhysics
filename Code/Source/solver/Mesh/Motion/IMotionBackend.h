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

#ifndef SVMP_MESH_MOTION_IMOTIONBACKEND_H
#define SVMP_MESH_MOTION_IMOTIONBACKEND_H

/**
 * @file IMotionBackend.h
 * @brief FE/solver-agnostic dependency injection interface for mesh motion.
 *
 * This interface lives in the Mesh library so MeshMotion can depend on it
 * without pulling in FE or linear algebra backends. Concrete implementations
 * are expected to live in higher-level libraries (e.g., future Physics
 * libraries) and are injected at runtime.
 */

#include "../Core/MeshTypes.h"
#include "MotionConfig.h"

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace svmp {

class DistributedMesh;
// Phase 5 (UNIFY_MESH): prefer the unified runtime mesh type name.
// In the Mesh library, `Mesh` is currently an alias of `DistributedMesh`.
using Mesh = DistributedMesh;

namespace motion {

/**
 * @brief Non-owning view of a mesh-attached motion field.
 *
 * The underlying storage is assumed to be vertex-major with a fixed number
 * of components per vertex (e.g., nsd).
 */
struct MotionFieldView {
  real_t* data = nullptr;
  size_t  n_entities = 0;
  size_t  components = 0;

  bool valid() const noexcept { return (data != nullptr) && (n_entities > 0) && (components > 0); }
};

/**
 * @brief Dirichlet boundary condition for mesh motion.
 *
 * The displacement returned by @p value is interpreted as an increment for
 * the current advance step.
 */
struct MotionDirichletBC {
  label_t boundary_label = INVALID_LABEL; ///< Boundary face marker; INVALID_LABEL => all boundary faces.

  /// value(x, dt, step_scale) -> displacement increment for the current solve/substep
  std::function<std::array<real_t, 3>(const std::array<real_t, 3>&, double, double)> value;
};

/**
 * @brief Inputs to a mesh-motion solve (typically: compute displacement and optionally velocity).
 */
struct MotionSolveRequest {
  Mesh& mesh;
  const MotionConfig& config; ///< Mesh motion configuration.

  double dt = 0.0;         ///< Caller time step size (typically the full step requested by MeshMotion::advance()).
  double step_scale = 1.0; ///< Fraction of dt used for the current solve (MeshMotion may call solve() multiple times per advance()).

  /// Explicit geometry configuration to use for coordinate queries during the solve.
  Configuration geometry_config = Configuration::Reference;

  MotionFieldView displacement; ///< Output (required): vertex displacement increments.
  MotionFieldView velocity;     ///< Output (optional): vertex velocities; backend may leave unset.

  /// Optional list of Dirichlet BCs (may be nullptr or empty).
  const std::vector<MotionDirichletBC>* dirichlet_bcs = nullptr;
};

/**
 * @brief Result of a mesh-motion solve.
 */
struct MotionSolveResult {
  bool success = true;
  bool wrote_velocity = false;
  std::string message;
};

/**
 * @brief Abstract backend interface for mesh motion.
 */
class IMotionBackend {
public:
  virtual ~IMotionBackend() = default;

  [[nodiscard]] virtual const char* name() const noexcept = 0;

  /**
   * @brief Solve for mesh motion fields for a single advance step.
   *
   * Implementations should write displacement into @p request.displacement.
   */
  virtual MotionSolveResult solve(const MotionSolveRequest& request) = 0;
};

} // namespace motion
} // namespace svmp

#endif // SVMP_MESH_MOTION_IMOTIONBACKEND_H
