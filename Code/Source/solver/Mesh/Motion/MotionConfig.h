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

#ifndef SVMP_MOTION_CONFIG_H
#define SVMP_MOTION_CONFIG_H

/**
 * @file MotionConfig.h
 * @brief Configuration structures for mesh motion models.
 *
 * MotionConfig defines Mesh-owned orchestration controls for mesh motion:
 * substepping limits and Mesh-side quality guards. Motion models, weak forms,
 * and solver options are intentionally out of scope for the Mesh library and
 * belong in injected backends (e.g., future Physics libraries).
 */

#include <limits>

namespace svmp {
namespace motion {

/// @brief Configuration parameters for mesh motion.
struct MotionConfig {
  /// Maximum step scaling factor applied within a single call to advance().
  /// Interpreted by MeshMotion as a fraction of dt used per substep; values
  /// in (0,1] enable sub-stepping for robustness. Values > 1.0 are reserved
  /// for future step-growth policies and are currently treated as 1.0.
  double max_step_scale = 1.0;

  /// Maximum number of sub-steps permitted in advance() when quality guards
  /// request step-size reduction.
  int max_substeps = 10;

  // ---- Motion quality guards (Mesh-side) ----

  /// Enable mesh-quality checks and step backtracking inside MeshMotion.
  bool enable_quality_guard = true;

  /// If true, MeshMotion will backtrack when quality thresholds are violated.
  /// If false, threshold violations are accepted but may reduce subsequent
  /// substep sizes.
  bool enforce_quality_thresholds = false;

  /// Minimum acceptable Jacobian quality (MeshQuality::Metric::Jacobian) before
  /// suggesting step-size reduction. Values are in [0,1]; 0 disables the guard.
  double quality_min_jacobian = 0.0;

  /// Minimum acceptable interior angle in degrees before suggesting step-size reduction.
  /// A value <= 0 disables the guard.
  double quality_min_angle_deg = 0.0;

  /// Maximum acceptable skewness (0 is best, ~1 is worst) before suggesting step-size reduction.
  /// Use infinity to disable the guard.
  double quality_max_skewness = std::numeric_limits<double>::infinity();
};

} // namespace motion
} // namespace svmp

#endif // SVMP_MOTION_CONFIG_H
