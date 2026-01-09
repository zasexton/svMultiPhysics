#ifndef SVMP_PHYSICS_CORE_PARAMETER_SCHEMA_H
#define SVMP_PHYSICS_CORE_PARAMETER_SCHEMA_H

/**
 * @file ParameterSchema.h
 * @brief Physics-side parameter schema and validation helper
 *
 * This is a thin wrapper around FE::systems::ParameterRegistry that allows
 * Physics modules to declare required/optional parameters and validate a
 * SystemStateView before assembly/solve.
 */

#include "FE/Core/ParameterValue.h"
#include "FE/Systems/ParameterRegistry.h"

#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {
struct SystemStateView;
} // namespace systems
} // namespace FE

namespace Physics {

class ParameterSchema {
public:
    ParameterSchema() = default;

    void clear();

    void add(FE::params::Spec spec, std::string source = {});

    void addAll(const std::vector<FE::params::Spec>& specs, std::string source = {});

    [[nodiscard]] const std::vector<FE::params::Spec>& specs() const noexcept;

    [[nodiscard]] const FE::params::Spec* find(std::string_view key) const noexcept;

    void validate(const FE::systems::SystemStateView& state) const;

private:
    FE::systems::ParameterRegistry registry_{};
};

} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_CORE_PARAMETER_SCHEMA_H

