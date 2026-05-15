#include "Application/Core/ActiveDomainOutput.h"

#include <algorithm>
#include <stdexcept>

namespace application {
namespace core {

std::vector<double> collectWetVolumeFractions(
    std::size_t n_cells,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules)
{
  std::vector<double> wet_fraction(n_cells, 0.0);
  for (const auto* rule : rules) {
    if (rule == nullptr) {
      continue;
    }
    const auto cell = rule->provenance.parent_entity;
    if (cell < 0 || static_cast<std::size_t>(cell) >= wet_fraction.size()) {
      continue;
    }
    auto& fraction = wet_fraction[static_cast<std::size_t>(cell)];
    fraction = std::clamp(
        fraction + static_cast<double>(rule->volume_fraction),
        0.0,
        1.0);
  }
  return wet_fraction;
}

std::size_t writeWetVolumeFractionField(
    svmp::Mesh& mesh,
    const std::string& field_name,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules)
{
  svmp::FieldHandle handle;
  if (mesh.has_field(svmp::EntityKind::Volume, field_name)) {
    handle = mesh.field_handle(svmp::EntityKind::Volume, field_name);
    if (mesh.field_type(handle) != svmp::FieldScalarType::Float64 ||
        mesh.field_components(handle) != 1u) {
      mesh.remove_field(handle);
      handle = mesh.attach_field(svmp::EntityKind::Volume,
                                 field_name,
                                 svmp::FieldScalarType::Float64,
                                 1u);
    }
  } else {
    handle = mesh.attach_field(svmp::EntityKind::Volume,
                               field_name,
                               svmp::FieldScalarType::Float64,
                               1u);
  }

  auto* data = static_cast<double*>(mesh.field_data(handle));
  if (data == nullptr) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Failed to allocate VTK cell field '" +
        field_name + "'.");
  }

  const auto wet_fraction = collectWetVolumeFractions(mesh.n_cells(), rules);
  std::copy(wet_fraction.begin(), wet_fraction.end(), data);
  return 1u;
}

} // namespace core
} // namespace application
