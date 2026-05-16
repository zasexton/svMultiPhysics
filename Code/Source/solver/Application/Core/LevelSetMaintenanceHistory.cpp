#include "Application/Core/LevelSetMaintenanceHistory.h"

#include "FE/Systems/FESystem.h"

#include <algorithm>
#include <stdexcept>

namespace application {
namespace core {

std::size_t copyFieldDofsIntoFeOrderedSolution(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId field,
    std::span<const svmp::FE::Real> source,
    std::vector<svmp::FE::Real>& target)
{
  if (field == svmp::FE::INVALID_FIELD_ID) {
    throw std::invalid_argument(
        "copyFieldDofsIntoFeOrderedSolution: invalid field id");
  }
  if (source.size() != target.size()) {
    throw std::invalid_argument(
        "copyFieldDofsIntoFeOrderedSolution: source and target sizes differ");
  }

  const auto& field_dofs = system.fieldDofHandler(field);
  const auto offset = system.fieldDofOffset(field);
  const auto n_dofs = field_dofs.getNumDofs();
  if (offset < 0 || n_dofs < 0 ||
      static_cast<std::size_t>(offset + n_dofs) > source.size()) {
    throw std::invalid_argument(
        "copyFieldDofsIntoFeOrderedSolution: field range is outside the FE-ordered solution");
  }

  const auto begin = static_cast<std::size_t>(offset);
  const auto count = static_cast<std::size_t>(n_dofs);
  std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(begin),
              count,
              target.begin() + static_cast<std::ptrdiff_t>(begin));
  return count;
}

} // namespace core
} // namespace application

