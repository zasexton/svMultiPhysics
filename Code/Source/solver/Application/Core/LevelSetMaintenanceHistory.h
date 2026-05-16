#pragma once

#include "FE/Core/Types.h"

#include <cstddef>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {
class FESystem;
} // namespace systems
} // namespace FE
} // namespace svmp

namespace application {
namespace core {

[[nodiscard]] std::size_t copyFieldDofsIntoFeOrderedSolution(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId field,
    std::span<const svmp::FE::Real> source,
    std::vector<svmp::FE::Real>& target);

} // namespace core
} // namespace application

