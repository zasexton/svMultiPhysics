#ifndef SVMP_FE_FORMS_JIT_JIT_SPECIALIZATION_H
#define SVMP_FE_FORMS_JIT_JIT_SPECIALIZATION_H

/**
 * @file JITSpecialization.h
 * @brief Compile-time size specialization for LLVM JIT kernels
 *
 * This header contains no LLVM dependencies.
 */

#include <cstdint>
#include <optional>

namespace svmp {
namespace FE {
namespace forms {

enum class IntegralDomain : std::uint8_t;

namespace jit {

/**
 * @brief Optional compile-time specialization for kernel loop trip counts.
 *
 * The specialization applies to the kernel group matching @p domain. For Cell
 * and Boundary kernels, only the "minus" fields are used (single-side
 * integration). For face kernels, both minus and plus may be provided.
 */
struct JITCompileSpecialization {
    IntegralDomain domain{};

    std::optional<std::uint32_t> n_qpts_minus{};
    std::optional<std::uint32_t> n_test_dofs_minus{};
    std::optional<std::uint32_t> n_trial_dofs_minus{};

    std::optional<std::uint32_t> n_qpts_plus{};
    std::optional<std::uint32_t> n_test_dofs_plus{};
    std::optional<std::uint32_t> n_trial_dofs_plus{};
};

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_JIT_SPECIALIZATION_H

