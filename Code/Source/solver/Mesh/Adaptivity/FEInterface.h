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

#ifndef SVMP_ADAPTIVITY_FE_INTERFACE_H
#define SVMP_ADAPTIVITY_FE_INTERFACE_H

/**
 * @file FEInterface.h
 * @brief Optional Mesh ↔ FE hook interface for full adaptivity support
 *
 * The Mesh library owns adaptivity of mesh topology/geometry and remains
 * independent of the FE library. To support:
 * - true PDE residual-based error estimation,
 * - DOF-level hanging constraints,
 * - solver-level transfer (projection/conservative remap),
 * - p>2 curved geometry refinement (high-order embedding),
 * the application can inject an FE-implemented interface into Mesh adaptivity.
 *
 * This header defines the interface in Mesh. The FE library implements it
 * without Mesh depending on FE headers (dependency inversion).
 */

#include "HighOrderEmbedding.h"
#include "Options.h"
#include "RefinementDelta.h"

#include <vector>

namespace svmp {

class MeshBase;

/**
 * @brief FE hook interface used by Mesh adaptivity (optional).
 *
 * Mesh will operate without an FE interface:
 * - For p<=2 geometry refinement, Mesh can compute isoparametric embeddings internally.
 * - For p>2, Mesh requires `get_high_order_embedding`.
 */
class AdaptivityFEInterface : public HighOrderEmbeddingProvider {
public:
  ~AdaptivityFEInterface() override = default;

  /**
   * @brief Optional: compute element-wise error indicators using FE/operator data.
   *
   * Returning an empty vector indicates "not provided" and Mesh will fall back
   * to its built-in estimators.
   */
  virtual std::vector<double> compute_error_indicators(
      const MeshBase& mesh,
      const AdaptivityOptions& options) const {
    (void)mesh;
    (void)options;
    return {};
  }

  /**
   * @brief Optional: notification that a mesh adaptivity step completed.
   *
   * FE implementations can rebuild DOF handlers, construct solver constraints,
   * and transfer solution vectors using `delta` (GID-based).
   */
  virtual void on_mesh_adapted(
      const MeshBase& old_mesh,
      const MeshBase& new_mesh,
      const RefinementDelta& delta,
      const AdaptivityOptions& options) {
    (void)old_mesh;
    (void)new_mesh;
    (void)delta;
    (void)options;
  }
};

} // namespace svmp

#endif // SVMP_ADAPTIVITY_FE_INTERFACE_H

