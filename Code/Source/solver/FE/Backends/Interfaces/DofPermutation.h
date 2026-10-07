/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_DOF_PERMUTATION_H
#define SVMP_FE_BACKENDS_DOF_PERMUTATION_H

#include "Core/Types.h"

#include <cstdint>
#include <cstring>
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
    // Optional backend-row ownership metadata. When present, owner_rank[backend_dof]
    // gives the MPI rank that owns the backend row for that DOF.
    std::vector<int> owner_rank{};
    // Optional partition-independent key of every backend node (indexed by
    // backend node id, i.e. backend_dof / dof_per_node): equal for the same
    // mesh node on every partition and distinct for distinct nodes.  Used by
    // preconditioners whose construction must not depend on the partition.
    std::vector<std::uint64_t> node_key{};

    [[nodiscard]] bool empty() const noexcept { return forward.empty() && inverse.empty(); }
};

/// Partition-independent node key from vertex coordinates: the bit patterns of
/// the coordinates mixed by the splitmix64 finalizer (+0 and -0 are equal).
[[nodiscard]] inline std::uint64_t nodeKeyFromCoordinates(double x, double y, double z) noexcept
{
    auto mix = [](std::uint64_t k) {
        std::uint64_t v = k + 0x9E3779B97F4A7C15ULL;
        v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
        v = (v ^ (v >> 27)) * 0x94D049BB133111EBULL;
        return v ^ (v >> 31);
    };
    auto bits = [](double v) {
        if (v == 0.0) {
            v = 0.0;
        }
        std::uint64_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        return u;
    };
    std::uint64_t h = mix(bits(x));
    h = mix(h ^ bits(y));
    h = mix(h ^ bits(z));
    return h;
}

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_DOF_PERMUTATION_H
