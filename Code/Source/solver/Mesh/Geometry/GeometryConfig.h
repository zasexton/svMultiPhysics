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

#ifndef SVMP_GEOMETRY_CONFIG_H
#define SVMP_GEOMETRY_CONFIG_H

#include "../Core/MeshTypes.h"
#include <cstddef>

namespace svmp {

/**
 * @brief Global numeric policies and tolerances for geometry computations.
 *
 * These values centralize small epsilons used to guard degeneracies
 * (zero length/area/volume) and to stabilize normalization.
 *
 * Keep values conservative; expose setters if runtime tuning is needed later.
 */
struct GeometryConfig {
  // Epsilon for length comparisons
  static constexpr real_t length_epsilon() noexcept { return static_cast<real_t>(1e-14); }

  // Epsilon for area/triangle area comparisons
  static constexpr real_t area_epsilon() noexcept { return static_cast<real_t>(1e-14); }

  // Epsilon for volume comparisons
  static constexpr real_t volume_epsilon() noexcept { return static_cast<real_t>(1e-16); }

  // Epsilon for vector normalization (avoid division by ~0)
  static constexpr real_t normal_epsilon() noexcept { return static_cast<real_t>(1e-20); }
};

} // namespace svmp

#endif // SVMP_GEOMETRY_CONFIG_H

