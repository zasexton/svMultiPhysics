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

#ifndef SVMP_MOTION_STATE_H
#define SVMP_MOTION_STATE_H

/**
 * @file MotionState.h
 * @brief Helpers for managing mesh motion state (coordinates, velocity).
 *
 * This header defines utilities for:
 *  - Backing up and restoring mesh coordinates around motion steps.
 *  - Updating mesh-velocity fields from displacement in a time step.
 *
 * All operations are expressed in terms of the Mesh local view (MeshBase)
 * and motion fields; no FE/DOF or solver-level concepts are introduced here.
 */

#include "../Core/MeshTypes.h"
#include "MotionFields.h"

#include <vector>

namespace svmp {

class MeshBase;

namespace motion {

/**
 * @brief Lightweight snapshot of mesh coordinate state for backtracking.
 *
 * Captures:
 *  - Whether a current-coordinate field existed.
 *  - The current-coordinate array (if present).
 *  - The active configuration (Reference or Current).
 *
 * Reference coordinates are intentionally not backed up; mesh motion
 * operates on top of a fixed reference configuration.
 */
struct MotionCoordinateBackup {
  std::vector<real_t> X_cur;
  bool has_current = false;
  Configuration active_config = Configuration::Reference;

  bool initialized = false;

  bool valid() const noexcept { return initialized; }
};

/**
 * @brief Save mesh coordinate state into a backup structure.
 *
 * The backup can be used later to restore coordinates if a motion
 * step is rejected by quality checks.
 *
 * @param mesh    Mesh whose state will be captured.
 * @param backup  Output backup data (overwritten).
 */
void save_coordinates(const MeshBase& mesh, MotionCoordinateBackup& backup);
void save_coordinates(const Mesh& mesh, MotionCoordinateBackup& backup);

/**
 * @brief Restore mesh coordinate state from a backup.
 *
 * If the backup indicates that no current-coordinate field existed,
 * this function clears any current coordinates and restores the
 * active configuration to reference.
 *
 * @param mesh    Mesh to restore.
 * @param backup  Backup to restore from.
 */
void restore_coordinates(MeshBase& mesh, const MotionCoordinateBackup& backup);
void restore_coordinates(Mesh& mesh, const MotionCoordinateBackup& backup);

/**
 * @brief Update mesh-velocity field from displacement over a time step.
 *
 * Interprets the displacement field as the change over a time step of
 * size @p dt and sets
 *
 *   w = u / dt
 *
 * per vertex and component. This is a simple finite-difference update;
 * the caller is responsible for ensuring that the displacement field
 * indeed represents the increment over @p dt.
 *
 * @param mesh Mesh on which the motion fields live.
 * @param hnd  Handles to displacement and velocity fields.
 * @param dt   Time step size (must be positive).
 */
void update_velocity_from_displacement(MeshBase& mesh,
                                       const MotionFieldHandles& hnd,
                                       real_t dt);
void update_velocity_from_displacement(Mesh& mesh,
                                       const MotionFieldHandles& hnd,
                                       real_t dt);

} // namespace motion
} // namespace svmp

#endif // SVMP_MOTION_STATE_H
