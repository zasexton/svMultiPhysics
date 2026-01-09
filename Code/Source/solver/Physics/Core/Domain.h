#ifndef SVMP_PHYSICS_CORE_DOMAIN_H
#define SVMP_PHYSICS_CORE_DOMAIN_H

/**
 * @file Domain.h
 * @brief Lightweight domain/marker conventions for Physics modules
 *
 * Physics modules should treat boundary/subdomain IDs as data (mesh labels),
 * not hard-coded integers. This header provides shared conventions to keep
 * APIs consistent across formulations.
 */

#include <vector>

namespace svmp {
namespace Physics {

// Convention aligned with FE/Forms:
// - boundary marker < 0 means "all boundary markers" (ds(-1))
inline constexpr int kAllBoundaryMarkers = -1;

inline constexpr int kAllSubdomains = -1;

struct MarkerSet {
    std::vector<int> markers{};

    [[nodiscard]] bool isAll() const noexcept { return markers.empty() || (markers.size() == 1u && markers[0] < 0); }

    [[nodiscard]] bool contains(int marker) const noexcept
    {
        if (isAll()) return true;
        for (int m : markers) {
            if (m == marker) return true;
        }
        return false;
    }
};

} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_CORE_DOMAIN_H

