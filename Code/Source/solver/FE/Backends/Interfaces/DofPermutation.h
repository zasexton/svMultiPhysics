/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_DOF_PERMUTATION_H
#define SVMP_FE_BACKENDS_DOF_PERMUTATION_H

#include "Core/Types.h"

#include <vector>

namespace svmp {
namespace FE {
namespace backends {

/**
 * @brief Global DOF permutation between FE ordering and backend ordering.
 *
 * `forward[fe] = backend` and `inverse[backend] = fe`.
 *
 * When empty, the identity permutation is implied.
 */
struct DofPermutation {
    std::vector<GlobalIndex> forward{};
    std::vector<GlobalIndex> inverse{};

    [[nodiscard]] bool empty() const noexcept { return forward.empty() && inverse.empty(); }
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_DOF_PERMUTATION_H

