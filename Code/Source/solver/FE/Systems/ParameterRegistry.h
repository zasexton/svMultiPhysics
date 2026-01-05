/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_PARAMETERREGISTRY_H
#define SVMP_FE_SYSTEMS_PARAMETERREGISTRY_H

#include "Core/FEException.h"
#include "Core/ParameterValue.h"
#include "Systems/SystemState.h"

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Registry of parameter requirements (key + type + defaults)
 *
 * This is a lightweight validation layer meant to keep parameter contracts
 * explicit and physics-agnostic. It can be populated from AssemblyKernels,
 * Forms ConstitutiveModels, and GlobalKernels during system setup.
 */
class ParameterRegistry {
public:
    ParameterRegistry() = default;

    void clear();

    void add(params::Spec spec, std::string source = {});
    void addAll(const std::vector<params::Spec>& specs, std::string source = {});

    [[nodiscard]] const std::vector<params::Spec>& specs() const noexcept { return specs_; }
    [[nodiscard]] const params::Spec* find(std::string_view key) const noexcept;

    /**
     * @brief Validate that the provided SystemStateView can satisfy required parameters.
     *
     * This checks presence and (when declared) type compatibility. Defaults in the
     * registry satisfy missing values.
     */
    void validate(const SystemStateView& state) const;

    /**
     * @brief Create a parameter getter that falls back to registry defaults.
     *
     * The returned callable is safe to use only while the referenced @p state
     * remains alive.
     */
    [[nodiscard]] std::function<std::optional<params::Value>(std::string_view)>
    makeParamGetter(const SystemStateView& state) const;

    /**
     * @brief Create a Real-only parameter getter that falls back to registry defaults.
     */
    [[nodiscard]] std::function<std::optional<Real>(std::string_view)>
    makeRealGetter(const SystemStateView& state) const;

private:
    struct Entry {
        params::Spec spec{};
        std::string source{};
    };

    static void validateDefaultType(const params::Spec& spec);
    static void mergeInto(params::Spec& dst, const params::Spec& src);

    std::vector<params::Spec> specs_{};
    std::unordered_map<std::string, Entry> by_key_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_PARAMETERREGISTRY_H

