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

#include "AnisotropicAdaptivity.h"
#include "../Core/MeshBase.h"

namespace svmp {

AnisotropicErrorEstimator::AnisotropicErrorEstimator(const Config& config)
    : config_(config) {}

std::vector<double> AnisotropicErrorEstimator::estimate(const MeshBase& mesh,
                                                        const MeshFields* fields,
                                                        const AdaptivityOptions& options) const {
  (void)fields;
  (void)options;
  const size_t n = mesh.n_cells();
  std::vector<double> indicators(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    indicators[i] = mesh.cell_measure(static_cast<index_t>(i));
  }
  return indicators;
}

AnisotropicMarker::AnisotropicMarker(const Config& config)
    : config_(config) {}

std::vector<MarkType> AnisotropicMarker::mark(const std::vector<double>& indicators,
                                              const MeshBase& mesh,
                                              const AdaptivityOptions& options) const {
  (void)mesh;
  std::vector<MarkType> marks(indicators.size(), MarkType::NONE);
  for (size_t i = 0; i < indicators.size(); ++i) {
    if (options.enable_refinement && indicators[i] >= options.refine_threshold) {
      marks[i] = MarkType::REFINE;
    } else if (options.enable_coarsening && indicators[i] <= options.coarsen_threshold) {
      marks[i] = MarkType::COARSEN;
    }
  }
  return marks;
}

AnisotropicAdaptivityManager::AnisotropicAdaptivityManager(const Config& config)
    : config_(config) {}

AdaptivityResult AnisotropicAdaptivityManager::adapt_anisotropic(MeshBase& mesh, MeshFields* fields) {
  AdaptivityManager manager(config_.options);
  return manager.adapt(mesh, fields);
}

} // namespace svmp

