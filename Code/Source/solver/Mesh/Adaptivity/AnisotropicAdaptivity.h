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

#ifndef SVMP_ANISOTROPIC_ADAPTIVITY_H
#define SVMP_ANISOTROPIC_ADAPTIVITY_H

#include "AdaptivityManager.h"
#include "ErrorEstimator.h"
#include "Marker.h"
#include "Options.h"

namespace svmp {

class MeshBase;
class MeshFields;

/**
 * @brief Placeholder metric tensor for anisotropic refinement.
 */
struct MetricTensor {
  bool is_3d = false;
};

/**
 * @brief Placeholder anisotropic mark information.
 */
struct AnisotropicMark {
  size_t element_id = 0;
  MarkType base_mark = MarkType::NONE;
};

/**
 * @brief Minimal anisotropic error estimator stub.
 *
 * Provides a MeshBase-compatible implementation of the ErrorEstimator interface so that
 * builds with Eigen enabled compile cleanly. Full anisotropic error estimation will be
 * implemented once directional refinement rules are integrated into the mesh core.
 */
class AnisotropicErrorEstimator final : public ErrorEstimator {
public:
  struct Config {
    int _unused = 0;
  };

  AnisotropicErrorEstimator() : AnisotropicErrorEstimator(Config{}) {}
  explicit AnisotropicErrorEstimator(const Config& config);

  std::vector<double> estimate(const MeshBase& mesh,
                               const MeshFields* fields,
                               const AdaptivityOptions& options) const override;

  std::string name() const override { return "AnisotropicErrorStub"; }

private:
  Config config_;
};

/**
 * @brief Minimal anisotropic marker stub.
 */
class AnisotropicMarker final : public Marker {
public:
  struct Config {
    int _unused = 0;
  };

  AnisotropicMarker() : AnisotropicMarker(Config{}) {}
  explicit AnisotropicMarker(const Config& config);

  std::vector<MarkType> mark(const std::vector<double>& indicators,
                             const MeshBase& mesh,
                             const AdaptivityOptions& options) const override;

  std::string name() const override { return "AnisotropicMarkerStub"; }

private:
  Config config_;
};

/**
 * @brief Minimal anisotropic adaptivity manager stub.
 */
class AnisotropicAdaptivityManager {
public:
  struct Config {
    AdaptivityOptions options;
  };

  AnisotropicAdaptivityManager() : AnisotropicAdaptivityManager(Config{}) {}
  explicit AnisotropicAdaptivityManager(const Config& config);

  AdaptivityResult adapt_anisotropic(MeshBase& mesh, MeshFields* fields = nullptr);

private:
  Config config_;
};

} // namespace svmp

#endif // SVMP_ANISOTROPIC_ADAPTIVITY_H

