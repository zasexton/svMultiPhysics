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

#include "MotionQuality.h"

#include "../Geometry/MeshQuality.h"
#include "../Core/MeshBase.h"
#include "../Core/DistributedMesh.h"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace motion {

MotionQualityReport evaluate_motion_quality(const MeshBase& mesh,
                                            Configuration cfg)
{
  MotionQualityReport report{};

  if (mesh.n_cells() == 0) {
    // Empty mesh: keep defaults and mark as non-inverted.
    report.has_inverted_cells = false;
    return report;
  }

  using MQ = MeshQuality;

  // Jacobian quality range
  {
    auto range = MQ::global_range(mesh, MQ::Metric::Jacobian, cfg);
    report.min_jacobian = range.first;
    report.max_jacobian = range.second;
  }

  // Scaled Jacobian range (if available for this mesh/family)
  {
    auto range = MQ::global_range(mesh, MQ::Metric::ScaledJacobian, cfg);
    report.min_scaled_jacobian = range.first;
    report.max_scaled_jacobian = range.second;
  }

  // Minimum angle (degrees)
  {
    auto range = MQ::global_range(mesh, MQ::Metric::MinAngle, cfg);
    report.min_angle_deg = range.first;
  }

  // Skewness (0 = perfect)
  {
    auto range = MQ::global_range(mesh, MQ::Metric::Skewness, cfg);
    report.max_skewness = range.second;
  }

  // Basic inversion check: negative Jacobian indicates inverted elements.
  report.has_inverted_cells = (report.min_jacobian <= 0.0);

  return report;
}

MotionQualityReport evaluate_motion_quality(const DistributedMesh& dmesh,
                                            Configuration cfg)
{
  const MeshBase& mesh = dmesh.local_mesh();

  MotionQualityReport report{};

  if (mesh.n_cells() == 0) {
    report.has_inverted_cells = false;
    return report;
  }

  using MQ = MeshQuality;

  real_t min_jacobian = std::numeric_limits<real_t>::infinity();
  real_t max_jacobian = -std::numeric_limits<real_t>::infinity();
  real_t min_scaled_jacobian = std::numeric_limits<real_t>::infinity();
  real_t max_scaled_jacobian = -std::numeric_limits<real_t>::infinity();
  real_t min_angle_deg = std::numeric_limits<real_t>::infinity();
  real_t max_skewness = -std::numeric_limits<real_t>::infinity();

  bool any_owned = false;

  for (size_t c = 0; c < mesh.n_cells(); ++c) {
    const auto cell = static_cast<index_t>(c);
    if (!dmesh.is_owned_cell(cell)) {
      continue;
    }
    any_owned = true;

    const real_t jac = MQ::compute(mesh, cell, MQ::Metric::Jacobian, cfg);
    min_jacobian = std::min(min_jacobian, jac);
    max_jacobian = std::max(max_jacobian, jac);

    const real_t sj = MQ::compute(mesh, cell, MQ::Metric::ScaledJacobian, cfg);
    min_scaled_jacobian = std::min(min_scaled_jacobian, sj);
    max_scaled_jacobian = std::max(max_scaled_jacobian, sj);

    const real_t ang = MQ::compute(mesh, cell, MQ::Metric::MinAngle, cfg);
    min_angle_deg = std::min(min_angle_deg, ang);

    const real_t skew = MQ::compute(mesh, cell, MQ::Metric::Skewness, cfg);
    max_skewness = std::max(max_skewness, skew);
  }

  // If this rank owns no cells, keep identity values so MPI reductions work.
  if (!any_owned) {
    min_jacobian = std::numeric_limits<real_t>::infinity();
    max_jacobian = -std::numeric_limits<real_t>::infinity();
    min_scaled_jacobian = std::numeric_limits<real_t>::infinity();
    max_scaled_jacobian = -std::numeric_limits<real_t>::infinity();
    min_angle_deg = std::numeric_limits<real_t>::infinity();
    max_skewness = -std::numeric_limits<real_t>::infinity();
  }

#ifdef MESH_HAS_MPI
  if (dmesh.world_size() > 1) {
    MPI_Comm comm = dmesh.mpi_comm();

    real_t tmp = min_jacobian;
    MPI_Allreduce(&tmp, &min_jacobian, 1, MPI_DOUBLE, MPI_MIN, comm);
    tmp = max_jacobian;
    MPI_Allreduce(&tmp, &max_jacobian, 1, MPI_DOUBLE, MPI_MAX, comm);

    tmp = min_scaled_jacobian;
    MPI_Allreduce(&tmp, &min_scaled_jacobian, 1, MPI_DOUBLE, MPI_MIN, comm);
    tmp = max_scaled_jacobian;
    MPI_Allreduce(&tmp, &max_scaled_jacobian, 1, MPI_DOUBLE, MPI_MAX, comm);

    tmp = min_angle_deg;
    MPI_Allreduce(&tmp, &min_angle_deg, 1, MPI_DOUBLE, MPI_MIN, comm);

    tmp = max_skewness;
    MPI_Allreduce(&tmp, &max_skewness, 1, MPI_DOUBLE, MPI_MAX, comm);
  }
#endif

  // Sanitize infinities for degenerate cases.
  if (!std::isfinite(min_jacobian)) {
    min_jacobian = 0.0;
  }
  if (!std::isfinite(max_jacobian)) {
    max_jacobian = 0.0;
  }
  if (!std::isfinite(min_scaled_jacobian)) {
    min_scaled_jacobian = 0.0;
  }
  if (!std::isfinite(max_scaled_jacobian)) {
    max_scaled_jacobian = 0.0;
  }
  if (!std::isfinite(min_angle_deg)) {
    min_angle_deg = 0.0;
  }
  if (!std::isfinite(max_skewness)) {
    max_skewness = 0.0;
  }

  report.min_jacobian = min_jacobian;
  report.max_jacobian = max_jacobian;
  report.min_scaled_jacobian = min_scaled_jacobian;
  report.max_scaled_jacobian = max_scaled_jacobian;
  report.min_angle_deg = min_angle_deg;
  report.max_skewness = max_skewness;
  report.has_inverted_cells = (report.min_jacobian <= 0.0);

  return report;
}

real_t suggested_step_scale(const MotionQualityReport& report,
                            real_t current_scale,
                            real_t detJ_min_threshold,
                            real_t min_angle_deg_threshold,
                            real_t max_skewness_threshold)
{
  if (current_scale <= 0.0) {
    return 0.0;
  }

  // Hard reject if any Jacobian is non-positive.
  if (report.has_inverted_cells || report.min_jacobian <= 0.0) {
    return 0.0;
  }

  real_t scale = current_scale;

  bool detj_violation   = report.min_jacobian < detJ_min_threshold;
  bool angle_violation  = report.min_angle_deg < min_angle_deg_threshold;
  bool skew_violation   = report.max_skewness > max_skewness_threshold;

  if (detj_violation || angle_violation || skew_violation) {
    // Conservative strategy: halve the step when any guard is violated.
    scale *= 0.5;
  }

  // Do not increase the scale beyond current.
  if (scale > current_scale) {
    scale = current_scale;
  }

  // Guard against denormals / extremely small values.
  if (scale < 1e-12) {
    scale = 0.0;
  }

  return scale;
}

} // namespace motion
} // namespace svmp
