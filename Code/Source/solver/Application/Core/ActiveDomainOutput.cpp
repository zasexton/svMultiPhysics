#include "Application/Core/ActiveDomainOutput.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/Quadrature/QuadratureFactory.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {

[[nodiscard]] bool isFullSideVolumeRule(
    const svmp::FE::geometry::CutQuadratureRule& rule) noexcept
{
  using svmp::FE::Real;
  if (rule.kind != svmp::FE::geometry::CutQuadratureKind::Volume ||
      !std::isfinite(rule.measure) ||
      !std::isfinite(rule.parent_measure) ||
      !std::isfinite(rule.volume_fraction) ||
      rule.parent_measure <= Real{0.0}) {
    return false;
  }

  const Real fraction_tol = std::numeric_limits<Real>::epsilon() * Real{128.0};
  const Real measure_scale = std::max<Real>(Real{1.0}, std::abs(rule.parent_measure));
  const Real measure_tol =
      std::numeric_limits<Real>::epsilon() * Real{128.0} * measure_scale;
  return std::abs(rule.volume_fraction - Real{1.0}) <= fraction_tol &&
         std::abs(rule.measure - rule.parent_measure) <= measure_tol;
}

[[nodiscard]] std::shared_ptr<svmp::FE::geometry::GeometryMapping>
makeCellMapping(
    const svmp::FE::assembly::IMeshAccess& mesh,
    svmp::FE::GlobalIndex cell_id)
{
  std::vector<std::array<svmp::FE::Real, 3>> cell_coords;
  mesh.getCellCoordinates(cell_id, cell_coords);

  std::vector<svmp::FE::math::Vector<svmp::FE::Real, 3>> node_coords;
  node_coords.reserve(cell_coords.size());
  for (const auto& coord : cell_coords) {
    node_coords.push_back(
        svmp::FE::math::Vector<svmp::FE::Real, 3>{
            coord[0], coord[1], coord[2]});
  }

  svmp::FE::geometry::MappingRequest request;
  request.element_type = mesh.getCellType(cell_id);
  request.geometry_order = mesh.getCellGeometryOrder(cell_id);
  request.use_affine = request.geometry_order <= 1;
  return svmp::FE::geometry::MappingFactory::create(request, node_coords);
}

[[nodiscard]] svmp::FE::Real physicalCellMeasure(
    const svmp::FE::assembly::IMeshAccess& mesh,
    svmp::FE::GlobalIndex cell_id)
{
  const auto mapping = makeCellMapping(mesh, cell_id);
  const int geometry_order = std::max(1, mesh.getCellGeometryOrder(cell_id));
  const auto quadrature =
      svmp::FE::quadrature::QuadratureFactory::create(
          mesh.getCellType(cell_id),
          std::max(2, 2 * geometry_order));

  svmp::FE::Real measure{0.0};
  for (std::size_t q = 0; q < quadrature->num_points(); ++q) {
    const auto xi = quadrature->point(q);
    const auto det_j = mapping->jacobian_determinant(xi);
    if (!std::isfinite(det_j) || !std::isfinite(quadrature->weight(q))) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Non-finite parent-cell Jacobian while measuring physical cut volume.");
    }
    measure += quadrature->weight(q) * std::abs(det_j);
  }
  return measure;
}

[[nodiscard]] svmp::FE::Real physicalCutVolumeRuleMeasure(
    const svmp::FE::assembly::IMeshAccess& mesh,
    const svmp::FE::geometry::CutQuadratureRule& rule)
{
  if (rule.kind != svmp::FE::geometry::CutQuadratureKind::Volume) {
    return svmp::FE::Real{0.0};
  }
  if (rule.frame == svmp::FE::geometry::CutGeometryFrame::Current) {
    if (!std::isfinite(rule.measure)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Non-finite current-frame cut-volume measure.");
    }
    return rule.measure;
  }

  const auto cell_id =
      static_cast<svmp::FE::GlobalIndex>(rule.provenance.parent_entity);
  if (cell_id < 0 || cell_id >= mesh.numCells()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Cut-volume parent cell is out of range while measuring physical wet volume.");
  }
  if (isFullSideVolumeRule(rule)) {
    return physicalCellMeasure(mesh, cell_id);
  }
  if (rule.points.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Partial cut-volume rule has no quadrature points.");
  }

  const auto mapping = makeCellMapping(mesh, cell_id);
  svmp::FE::Real measure{0.0};
  for (const auto& qp : rule.points) {
    const svmp::FE::math::Vector<svmp::FE::Real, 3> xi{
        qp.point[0], qp.point[1], qp.point[2]};
    const auto det_j = mapping->jacobian_determinant(xi);
    if (!std::isfinite(det_j) || !std::isfinite(qp.weight)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Non-finite cut quadrature weight or parent-cell Jacobian.");
    }
    measure += qp.weight * std::abs(det_j);
  }
  return measure;
}

} // namespace

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

CutVolumeMeasureSummary collectCutVolumeMeasures(
    const svmp::FE::assembly::IMeshAccess& mesh,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules)
{
  CutVolumeMeasureSummary summary;
  for (const auto* rule : rules) {
    if (rule == nullptr ||
        rule->kind != svmp::FE::geometry::CutQuadratureKind::Volume) {
      continue;
    }
    ++summary.rule_count;
    summary.reference_measure += rule->measure;
    try {
      summary.physical_measure += physicalCutVolumeRuleMeasure(mesh, *rule);
      ++summary.physical_rule_count;
    } catch (...) {
      ++summary.skipped_physical_rule_count;
    }
  }
  return summary;
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

WetVolumeDriftDiagnostic computeWetVolumeDrift(
    const std::string& key,
    svmp::FE::Real wet_volume,
    std::map<std::string, svmp::FE::Real>& initial_wet_volume_by_key)
{
  const auto [initial_it, inserted] =
      initial_wet_volume_by_key.try_emplace(key, wet_volume);
  (void)inserted;

  WetVolumeDriftDiagnostic diagnostic;
  diagnostic.initial_wet_volume = initial_it->second;
  diagnostic.wet_volume_drift = wet_volume - diagnostic.initial_wet_volume;
  diagnostic.relative_wet_volume_drift =
      std::abs(diagnostic.initial_wet_volume) > svmp::FE::Real{0.0}
          ? diagnostic.wet_volume_drift / diagnostic.initial_wet_volume
          : svmp::FE::Real{0.0};
  return diagnostic;
}

} // namespace core
} // namespace application
