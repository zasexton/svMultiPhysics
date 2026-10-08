/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TESTS_UNIT_TIMING_ASSERTIONS_H
#define SVMP_FE_TESTS_UNIT_TIMING_ASSERTIONS_H

#include <cstdlib>
#include <string_view>

namespace svmp {
namespace FE {
namespace test {

/// Wall-clock thresholds flake under parallel CTest on shared nodes, so unit
/// tests check them only when SVMP_ENABLE_TIMING_ASSERTIONS=1; the timed
/// computations and their correctness checks always run.
[[nodiscard]] inline bool timingAssertionsEnabled()
{
    const char* value = std::getenv("SVMP_ENABLE_TIMING_ASSERTIONS");
    return value != nullptr && std::string_view(value) == "1";
}

} // namespace test
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TESTS_UNIT_TIMING_ASSERTIONS_H
