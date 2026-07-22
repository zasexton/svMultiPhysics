#include "Application/Core/ActiveDomainOutput.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/MeshAccess.h"
#include "FE/Geometry/CutQuadratureMapping.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

struct MappedWetVolumeCellData {
  std::vector<double> fraction;
  std::vector<double> physical_wet_measure;
};

svmp::FieldHandle prepareScalarVolumeField(
    svmp::Mesh& mesh,
    const std::string& field_name)
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
  return handle;
}

void writeScalarVolumeField(
    svmp::Mesh& mesh,
    const std::string& field_name,
    const std::vector<double>& values)
{
  const auto handle = prepareScalarVolumeField(mesh, field_name);
  auto* data = static_cast<double*>(mesh.field_data(handle));
  if (data == nullptr) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Failed to allocate VTK cell field '" +
        field_name + "'.");
  }
  std::copy(values.begin(), values.end(), data);
}

MappedWetVolumeCellData collectMappedWetVolumeCellData(
    const svmp::FE::assembly::IMeshAccess& mesh,
    std::size_t n_cells,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules)
{
  auto reference_fraction =
      application::core::collectWetVolumeFractions(n_cells, rules);
  std::vector<svmp::FE::Real> wet_measure(n_cells, svmp::FE::Real{0.0});
  std::vector<svmp::FE::Real> parent_measure(n_cells, svmp::FE::Real{0.0});
  std::vector<bool> has_mapped_measure(n_cells, false);
  std::vector<bool> failed_mapped_measure(n_cells, false);

  for (const auto* rule : rules) {
    if (rule == nullptr ||
        rule->kind != svmp::FE::geometry::CutQuadratureKind::Volume) {
      continue;
    }
    const auto cell = rule->provenance.parent_entity;
    if (cell < 0 || static_cast<std::size_t>(cell) >= n_cells) {
      continue;
    }
    const auto index = static_cast<std::size_t>(cell);
    try {
      if (parent_measure[index] <= svmp::FE::Real{0.0}) {
        parent_measure[index] =
            svmp::FE::geometry::physicalCellMeasureFromMapping(mesh, cell);
      }
      wet_measure[index] +=
          svmp::FE::geometry::physicalCutQuadratureMeasure(mesh, *rule);
      has_mapped_measure[index] = true;
    } catch (...) {
      if (rule->provenance.free_surface_snapshot_revision_key != 0u ||
          rule->provenance.source_value_revision != 0u) {
        throw;
      }
      failed_mapped_measure[index] = true;
    }
  }

  for (std::size_t cell = 0; cell < n_cells; ++cell) {
    if (!has_mapped_measure[cell] || failed_mapped_measure[cell] ||
        parent_measure[cell] <= svmp::FE::Real{0.0} ||
        !std::isfinite(parent_measure[cell])) {
      if (parent_measure[cell] > svmp::FE::Real{0.0} &&
          std::isfinite(parent_measure[cell])) {
        wet_measure[cell] =
            static_cast<svmp::FE::Real>(reference_fraction[cell]) *
            parent_measure[cell];
      }
      continue;
    }
    reference_fraction[cell] = std::clamp(
        static_cast<double>(wet_measure[cell] / parent_measure[cell]),
        0.0,
        1.0);
  }

  std::vector<double> physical_wet_measure(n_cells, 0.0);
  for (std::size_t cell = 0; cell < n_cells; ++cell) {
    physical_wet_measure[cell] = static_cast<double>(wet_measure[cell]);
  }
  return {std::move(reference_fraction), std::move(physical_wet_measure)};
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
  bool found_revisioned_rule = false;
  bool found_unrevisioned_rule = false;
  for (const auto* rule : rules) {
    if (rule == nullptr ||
        rule->kind != svmp::FE::geometry::CutQuadratureKind::Volume) {
      continue;
    }
    const auto snapshot_revision_key =
        rule->provenance.free_surface_snapshot_revision_key;
    const auto source_value_revision =
        rule->provenance.source_value_revision;
    if ((snapshot_revision_key == 0u) != (source_value_revision == 0u)) {
      throw std::invalid_argument(
          "[svMultiPhysics::Application] Cut-volume measure rule has an "
          "incomplete free-surface snapshot revision.");
    }
    if (snapshot_revision_key != 0u) {
      found_revisioned_rule = true;
      if (summary.free_surface_snapshot_revision_key == 0u) {
        summary.free_surface_snapshot_revision_key = snapshot_revision_key;
        summary.source_value_revision = source_value_revision;
      } else if (summary.free_surface_snapshot_revision_key !=
                     snapshot_revision_key ||
                 summary.source_value_revision != source_value_revision) {
        throw std::invalid_argument(
            "[svMultiPhysics::Application] Cut-volume measure rules mix "
            "free-surface snapshot revisions.");
      }
    } else {
      found_unrevisioned_rule = true;
    }
  }
  if (found_revisioned_rule && found_unrevisioned_rule) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] Cut-volume measure rules mix "
        "revisioned and unrevisioned geometry.");
  }

  for (const auto* rule : rules) {
    if (rule == nullptr ||
        rule->kind != svmp::FE::geometry::CutQuadratureKind::Volume) {
      continue;
    }
    const auto parent_cell = static_cast<svmp::FE::GlobalIndex>(
        rule->provenance.parent_entity);
    if (parent_cell < 0 || parent_cell >= mesh.numCells() ||
        !mesh.isOwnedCell(parent_cell)) {
      continue;
    }
    if (found_revisioned_rule) {
      ++summary.revisioned_rule_count;
    }
    ++summary.rule_count;
    summary.reference_measure += rule->measure;
    try {
      summary.physical_measure +=
          svmp::FE::geometry::physicalCutQuadratureMeasure(mesh, *rule);
      ++summary.physical_rule_count;
    } catch (...) {
      if (found_revisioned_rule) {
        throw;
      }
      ++summary.skipped_physical_rule_count;
    }
  }
  return summary;
}

WetVolumeMeasureSelection selectWetVolumeForDrift(
    const CutVolumeMeasureSummary& summary)
{
  if (summary.revisioned_rule_count != 0u &&
      summary.skipped_physical_rule_count != 0u) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] Revision-bound cut-volume measure "
        "cannot fall back after a physical mapping failure.");
  }
  WetVolumeMeasureSelection selection;
  if (summary.skipped_physical_rule_count == 0u) {
    selection.wet_volume = summary.physical_measure;
    selection.frame = "physical";
  } else {
    selection.wet_volume = summary.reference_measure;
    selection.frame = "reference_fallback";
  }
  return selection;
}

std::size_t writeWetVolumeFractionField(
    svmp::Mesh& mesh,
    const std::string& field_name,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules,
    const std::string& measure_field_name)
{
  if (!measure_field_name.empty() && measure_field_name == field_name) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Wet volume fraction and measure fields must have different names.");
  }

  svmp::FE::assembly::MeshAccess mesh_access(mesh);
  const auto wet_volume =
      collectMappedWetVolumeCellData(mesh_access, mesh.n_cells(), rules);
  writeScalarVolumeField(mesh, field_name, wet_volume.fraction);
  if (measure_field_name.empty()) {
    return 1u;
  }
  writeScalarVolumeField(mesh,
                         measure_field_name,
                         wet_volume.physical_wet_measure);
  return 2u;
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
