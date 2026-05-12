#ifndef SVMP_PHYSICS_CORE_JIT_RUNTIME_POLICY_H
#define SVMP_PHYSICS_CORE_JIT_RUNTIME_POLICY_H

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/PhysicsJITPolicy.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

namespace svmp::Physics::core {

namespace detail {

inline std::string trim_copy(std::string s)
{
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

inline std::string lower_copy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

inline std::optional<bool> parse_bool_relaxed(std::string_view raw) noexcept
{
    const auto value = lower_copy(trim_copy(std::string(raw)));
    if (value == "true" || value == "1" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "false" || value == "0" || value == "no" || value == "off") {
        return false;
    }
    return std::nullopt;
}

inline std::optional<bool> lookup_param_bool(const ParameterMap& params, std::string_view key) noexcept
{
    const auto it = params.find(std::string(key));
    if (it == params.end() || !it->second.defined) {
        return std::nullopt;
    }
    return parse_bool_relaxed(it->second.value);
}

inline std::optional<bool> lookup_module_options_bool(std::string_view module_options) noexcept
{
    if (module_options.empty()) {
        return std::nullopt;
    }

    std::string normalized(module_options);
    for (char& ch : normalized) {
        if (ch == ';' || ch == '\n' || ch == '\t') {
            ch = ',';
        }
    }

    std::size_t start = 0;
    while (start < normalized.size()) {
        const std::size_t end = normalized.find(',', start);
        std::string token = trim_copy(normalized.substr(start, end - start));
        if (!token.empty()) {
            const std::size_t sep = token.find_first_of("=:");
            if (sep != std::string::npos) {
                const auto key = lower_copy(trim_copy(token.substr(0, sep)));
                const auto value = parse_bool_relaxed(token.substr(sep + 1));
                if (value.has_value() &&
                    (key == "jit" || key == "jit_enable" || key == "enable_jit")) {
                    return value;
                }
            }
        }

        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    return std::nullopt;
}

inline std::optional<bool> lookup_module_options_bool(std::string_view module_options,
                                                      std::initializer_list<std::string_view> keys) noexcept
{
    if (module_options.empty()) {
        return std::nullopt;
    }

    std::string normalized(module_options);
    for (char& ch : normalized) {
        if (ch == ';' || ch == '\n' || ch == '\t') {
            ch = ',';
        }
    }

    std::size_t start = 0;
    while (start < normalized.size()) {
        const std::size_t end = normalized.find(',', start);
        std::string token = trim_copy(normalized.substr(start, end - start));
        if (!token.empty()) {
            const std::size_t sep = token.find_first_of("=:");
            if (sep != std::string::npos) {
                const auto key = lower_copy(trim_copy(token.substr(0, sep)));
                const auto value = parse_bool_relaxed(token.substr(sep + 1));
                if (value.has_value()) {
                    for (const auto candidate : keys) {
                        if (key == candidate) {
                            return value;
                        }
                    }
                }
            }
        }

        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    return std::nullopt;
}

} // namespace detail

inline bool resolveOopJitEnable(const EquationModuleInput& input, bool default_enabled)
{
    if (const auto value = detail::lookup_module_options_bool(input.module_options,
                                                              {"jit", "jit_enable", "enable_jit"})) {
        return *value;
    }

    static constexpr std::string_view kParamKeys[] = {
        "Jit_enable",
        "Enable_jit",
        "Use_jit",
    };
    for (const auto key : kParamKeys) {
        if (const auto value = detail::lookup_param_bool(input.equation_params, key)) {
            return *value;
        }
    }

    static constexpr const char* kEnvKeys[] = {
        "SVMP_OOP_JIT_ENABLE",
        "SVMP_FE_JIT_ENABLE",
    };
    for (const char* key : kEnvKeys) {
        if (const char* env = std::getenv(key); env != nullptr) {
            if (const auto value = detail::parse_bool_relaxed(env)) {
                return *value;
            }
        }
    }

    return default_enabled;
}

inline bool resolveOopJitSpecializationEnable(const EquationModuleInput& input, bool default_enabled)
{
    if (const auto value = detail::lookup_module_options_bool(
            input.module_options,
            {"jit_specialization", "enable_jit_specialization", "jit.specialization", "jit_specialization_enable"})) {
        return *value;
    }

    static constexpr std::string_view kParamKeys[] = {
        "Jit_specialization_enable",
        "Enable_jit_specialization",
        "Use_jit_specialization",
    };
    for (const auto key : kParamKeys) {
        if (const auto value = detail::lookup_param_bool(input.equation_params, key)) {
            return *value;
        }
    }

    static constexpr const char* kEnvKeys[] = {
        "SVMP_OOP_JIT_SPECIALIZATION_ENABLE",
        "SVMP_FE_JIT_SPECIALIZATION_ENABLE",
    };
    for (const char* key : kEnvKeys) {
        if (const char* env = std::getenv(key); env != nullptr) {
            if (const auto value = detail::parse_bool_relaxed(env)) {
                return *value;
            }
        }
    }

    return default_enabled;
}

inline PhysicsJITPolicy resolveOopJitPolicy(const EquationModuleInput& input,
                                            PhysicsJITPolicy default_policy = {})
{
    PhysicsJITPolicy policy = default_policy;
    policy.enable = resolveOopJitEnable(input, default_policy.enable);
    policy.specialization =
        resolveOopJitSpecializationEnable(input, default_policy.specialization);
    return policy;
}

} // namespace svmp::Physics::core

#endif // SVMP_PHYSICS_CORE_JIT_RUNTIME_POLICY_H
