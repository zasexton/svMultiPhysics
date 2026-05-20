#ifndef SVMP_FE_FORMS_JIT_JIT_CACHE_KEY_H
#define SVMP_FE_FORMS_JIT_JIT_CACHE_KEY_H

/**
 * @file JITCacheKey.h
 * @brief Pure helpers for stable LLVM JIT kernel cache keys and symbols.
 */

#include "Forms/FormExpr.h"
#include "Forms/JIT/JITSpecialization.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace svmp {
namespace FE {
namespace forms {

enum class FormKind : std::uint8_t;
enum class IntegralDomain : std::uint8_t;

namespace jit {

inline constexpr std::uint64_t kCacheKeyFNVOffset = 14695981039346656037ULL;
inline constexpr std::uint64_t kCacheKeyFNVPrime = 1099511628211ULL;

// Increment this when cache-key inputs or generated kernel semantics change.
inline constexpr std::uint64_t kKernelCacheKeySchemaVersion = 8ULL;

struct KernelCacheKeyInputs {
    std::uint64_t cache_key_schema_version{kKernelCacheKeySchemaVersion};
    std::uint32_t abi_version{0};
    std::uint32_t abi_layout_revision{0};
    FormKind form_kind{};
    IntegralDomain domain{};
    int boundary_marker{-1};
    int interface_marker{-1};
    CutVolumeSide cut_volume_side{CutVolumeSide::Negative};
    std::uint64_t combined_ir_hash{0};
    std::uint64_t test_space_hash{0};
    std::uint64_t trial_space_hash{0};
    JITOptions jit_options{};
    std::string_view target_triple{};
    std::string_view data_layout{};
    std::string_view cpu_name{};
    std::string_view cpu_features{};
    std::string_view llvm_version{};
    std::uint64_t hardware_profile_hash{0};
    const JITCompileSpecialization* specialization{nullptr};
};

void mixCacheKey(std::uint64_t& h, std::uint64_t v) noexcept;

[[nodiscard]] std::uint64_t hashStringForCacheKey(std::string_view s) noexcept;
[[nodiscard]] std::uint64_t hashTensorOptionsForCacheKey(const TensorJITOptions& opt) noexcept;
[[nodiscard]] std::uint64_t hashSpecializationCodegenOptionsForCacheKey(
    const JITSpecializationOptions& opt) noexcept;

[[nodiscard]] std::uint64_t computeKernelCacheKey(const KernelCacheKeyInputs& inputs) noexcept;
[[nodiscard]] std::string cacheKeyToHex(std::uint64_t cache_key);
[[nodiscard]] std::string stableSymbolForKernel(std::uint64_t cache_key);

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_JIT_CACHE_KEY_H
