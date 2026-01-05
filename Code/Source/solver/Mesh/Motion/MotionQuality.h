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

#ifndef SVMP_MOTION_QUALITY_H
#define SVMP_MOTION_QUALITY_H

/**
 * @file MotionQuality.h
 * @brief Mesh-motion specific quality summaries and step advice.
 *
 * This header provides lightweight wrappers around MeshQuality to
 * evaluate mesh quality before/after a motion step and to suggest
 * simple step-scaling decisions. It is purely geometric and does
 * not depend on FE/DOF or solver concepts.
 */

#include "../Core/MeshTypes.h"

namespace svmp {

class MeshBase;
class DistributedMesh;
// Phase 5 (UNIFY_MESH): prefer the unified runtime mesh type name.
// In the Mesh library, `Mesh` is currently an alias of `DistributedMesh`.
using Mesh = DistributedMesh;

namespace motion {

/**
 * @brief Aggregate quality metrics relevant for mesh motion.
 *
 * The interpretation of "Jacobian" and "scaled Jacobian" follows
 * MeshQuality's conventions. Typically, positive Jacobians and
 * scaled-Jacobian values near 1 indicate good elements.
 */
struct MotionQualityReport {
  real_t min_jacobian        = 0.0;
  real_t max_jacobian        = 0.0;
  real_t min_scaled_jacobian = 0.0;
  real_t max_scaled_jacobian = 0.0;
  real_t min_angle_deg       = 0.0;
  real_t max_skewness        = 0.0;
  bool   has_inverted_cells  = false;
};

/**
 * @brief Evaluate motion-relevant quality metrics on a mesh.
 *
 * Uses MeshQuality to compute global min/max of:
 *  - Jacobian quality
 *  - Scaled Jacobian
 *  - Minimum angle
 *  - Skewness
 *
 * Note: this overload operates on a single MeshBase instance. For MPI meshes
 * prefer the Mesh overload, which filters to owned cells and performs MPI
 * reductions.
 *
 * The configuration @p cfg indicates whether quality is evaluated in
 * the reference or current configuration; for motion gating it is
 * typically Configuration::Current.
 *
 * @param mesh Mesh to evaluate.
 * @param cfg  Configuration (reference or current).
 * @return MotionQualityReport with aggregated metrics.
 */
MotionQualityReport evaluate_motion_quality(const MeshBase& mesh,
                                            Configuration cfg = Configuration::Current);

/**
 * @brief Evaluate motion-relevant quality metrics on a distributed mesh.
 *
 * The report is computed over owned cells only and, when MPI is enabled,
 * reduced across ranks to produce global min/max values.
 */
MotionQualityReport evaluate_motion_quality(const Mesh& mesh,
                                            Configuration cfg = Configuration::Current);

/**
 * @brief Simple step decision helper based on quality metrics.
 *
 * This function encapsulates a minimal policy:
 *  - If min_jacobian <= 0, the step is rejected (scale = 0).
 *  - If min_jacobian < detJ_min_threshold, or min_angle < min_angle_deg_threshold,
 *    or max_skewness > max_skewness_threshold, the step is accepted but the
 *    suggested scale is reduced (typically halved).
 *  - Otherwise, the current_scale is returned unchanged.
 *
 * The caller can interpret a suggested_scale of 0 as "reject step and
 * restore previous coordinates".
 *
 * @param report                  Quality report from evaluate_motion_quality.
 * @param current_scale           Current step scaling factor (e.g. 1.0).
 * @param detJ_min_threshold      Minimum acceptable Jacobian determinant (or proxy).
 * @param min_angle_deg_threshold Minimum acceptable interior angle in degrees.
 * @param max_skewness_threshold  Maximum acceptable skewness.
 * @return Suggested next step scale (0 => reject).
 */
real_t suggested_step_scale(const MotionQualityReport& report,
                            real_t current_scale,
                            real_t detJ_min_threshold,
                            real_t min_angle_deg_threshold,
                            real_t max_skewness_threshold);

} // namespace motion
} // namespace svmp

#endif // SVMP_MOTION_QUALITY_H
