/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_MULTIPLIERLOWERING_H
#define SVMP_FE_SYSTEMS_MULTIPLIERLOWERING_H

#include <cstdint>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief How the global multipliers of a system reach the linear algebra (D54)
 *
 * The form states the coupling; the lowering is chosen at setup.  InMatrix
 * assembles every global row and column into the matrix.  Bordered keeps the
 * global DOFs out of the matrix for backends with a fixed node layout
 * (FSILS): their rows and columns go to FESystem::borderedCoupling() and
 * Newton solves them with the bordered correction (LM2).
 */
enum class MultiplierLowering : std::uint8_t {
    Auto,     ///< In the matrix
    InMatrix, ///< Global rows and columns assembled into the matrix
    Bordered, ///< Global DOFs kept out of the matrix, bordered blocks solved by Newton
};

[[nodiscard]] constexpr const char* multiplierLoweringName(MultiplierLowering lowering) noexcept
{
    switch (lowering) {
        case MultiplierLowering::Auto:
            return "Auto";
        case MultiplierLowering::InMatrix:
            return "InMatrix";
        case MultiplierLowering::Bordered:
            return "Bordered";
    }
    return "unknown";
}

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_MULTIPLIERLOWERING_H
