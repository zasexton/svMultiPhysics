#ifndef SVMP_FE_CORE_ALIGNMENT_H
#define SVMP_FE_CORE_ALIGNMENT_H

/**
 * @file Alignment.h
 * @brief Global alignment constants used across FE modules.
 */

#include <cstddef>

namespace svmp {
namespace FE {

/// Preferred cache-line/SIMD alignment for performance-critical arrays.
inline constexpr std::size_t kFEPreferredAlignmentBytes = 64u;

} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CORE_ALIGNMENT_H

