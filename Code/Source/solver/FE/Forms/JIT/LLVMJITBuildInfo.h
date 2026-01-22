#ifndef SVMP_FE_FORMS_JIT_LLVM_JIT_BUILD_INFO_H
#define SVMP_FE_FORMS_JIT_LLVM_JIT_BUILD_INFO_H

/**
 * @file LLVMJITBuildInfo.h
 * @brief Small helpers describing whether the FE library was built with LLVM JIT enabled.
 *
 * This header contains no LLVM dependencies.
 */

#include <string>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

/// True when the FE build enabled LLVM JIT support (FE_ENABLE_LLVM_JIT=ON).
[[nodiscard]] bool llvmJITEnabled() noexcept;

/// LLVM version string when enabled, otherwise empty.
[[nodiscard]] std::string llvmVersionString();

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_LLVM_JIT_BUILD_INFO_H

