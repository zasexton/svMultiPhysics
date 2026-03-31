#ifndef SVMP_FE_CORE_KERNELTRACE_H
#define SVMP_FE_CORE_KERNELTRACE_H

/**
 * @file KernelTrace.h
 * @brief Shared tracing helpers for FE kernel lowering, JIT, and assembly
 */

#include "Core/Logger.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <string_view>

namespace svmp {
namespace FE {
namespace core {

enum class KernelTraceChannel : std::uint32_t {
    None = 0u,
    Selection = 1u << 0,
    Specialization = 1u << 1,
    Assembly = 1u << 2,
    Capabilities = 1u << 3,
    All = 0xFFFFu
};

inline constexpr KernelTraceChannel operator|(KernelTraceChannel a, KernelTraceChannel b) noexcept
{
    return static_cast<KernelTraceChannel>(static_cast<std::uint32_t>(a) |
                                           static_cast<std::uint32_t>(b));
}

inline constexpr bool hasFlag(KernelTraceChannel flags, KernelTraceChannel flag) noexcept
{
    return (static_cast<std::uint32_t>(flags) & static_cast<std::uint32_t>(flag)) != 0u;
}

[[nodiscard]] inline bool envEnabled(const char* name) noexcept
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    std::string v(value);
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(v == "0" || v == "false" || v == "off" || v == "no");
}

[[nodiscard]] inline KernelTraceChannel configuredKernelTraceChannels() noexcept
{
    static const KernelTraceChannel mask = [] {
        KernelTraceChannel out = KernelTraceChannel::None;

        if (envEnabled("SVMP_OOP_SOLVER_TRACE")) {
            out = out | KernelTraceChannel::Assembly;
        }
        if (envEnabled("SVMP_JIT_TRACE_SPECIALIZATION")) {
            out = out | KernelTraceChannel::Specialization;
        }

        const char* value = std::getenv("SVMP_FE_KERNEL_TRACE");
        if (value == nullptr || value[0] == '\0') {
            return out;
        }

        std::string raw(value);
        std::transform(raw.begin(), raw.end(), raw.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        std::string token;
        std::istringstream iss(raw);
        while (std::getline(iss, token, ',')) {
            token.erase(std::remove_if(token.begin(), token.end(),
                                       [](unsigned char c) { return std::isspace(c) != 0; }),
                        token.end());
            if (token.empty()) {
                continue;
            }
            if (token == "1" || token == "all" || token == "true" || token == "on") {
                return KernelTraceChannel::All;
            }
            if (token == "selection" || token == "path" || token == "paths") {
                out = out | KernelTraceChannel::Selection;
            } else if (token == "specialization" || token == "jit") {
                out = out | KernelTraceChannel::Specialization;
            } else if (token == "assembly" || token == "dispatch") {
                out = out | KernelTraceChannel::Assembly;
            } else if (token == "capability" || token == "capabilities" || token == "backend") {
                out = out | KernelTraceChannel::Capabilities;
            }
        }

        return out;
    }();

    return mask;
}

[[nodiscard]] inline bool kernelTraceEnabled(KernelTraceChannel channel) noexcept
{
    return hasFlag(configuredKernelTraceChannels(), channel);
}

inline void kernelTraceLog(KernelTraceChannel channel, const std::string& message)
{
    if (!kernelTraceEnabled(channel)) {
        return;
    }
    FE_LOG_INFO(message);
}

} // namespace core
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CORE_KERNELTRACE_H
