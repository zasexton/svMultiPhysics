#ifndef SVMP_FE_FORMS_JIT_LLVM_JIT_BUILD_INFO_H
#define SVMP_FE_FORMS_JIT_LLVM_JIT_BUILD_INFO_H

/**
 * @file LLVMJITBuildInfo.h
 * @brief Small helpers describing whether the FE library was built with LLVM JIT enabled.
 *
 * This header contains no LLVM dependencies.
 */

#include <string>
#include <string_view>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

/// True when the FE build enabled LLVM JIT support (FE_ENABLE_LLVM_JIT=ON).
[[nodiscard]] bool llvmJITEnabled() noexcept;

/// LLVM version string when enabled, otherwise empty.
[[nodiscard]] std::string llvmVersionString();

/**
 * @brief SHA-256 fingerprint of the compiled-kernel code generator sources
 *        (Forms/JIT and the kernel argument layout of Assembly/JIT) this
 *        binary was built from, as 64 lowercase hex digits.
 *
 * Part of every JIT object-cache key: binaries built from different
 * code-generator sources must not load each other's cached kernels for the
 * same IR, while binaries built from the same sources share them.
 */
[[nodiscard]] std::string jitCodegenFingerprint();
/// View of the same fingerprint with static lifetime.
[[nodiscard]] std::string_view jitCodegenFingerprintView() noexcept;

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_LLVM_JIT_BUILD_INFO_H

