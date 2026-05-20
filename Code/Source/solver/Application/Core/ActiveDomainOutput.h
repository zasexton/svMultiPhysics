#pragma once

#include "FE/Geometry/CutQuadrature.h"
#include "Mesh/Mesh.h"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
class IMeshAccess;
}
}
}

namespace application {
namespace core {

std::vector<double> collectWetVolumeFractions(
    std::size_t n_cells,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules);

struct CutVolumeMeasureSummary {
  svmp::FE::Real reference_measure{0.0};
  svmp::FE::Real physical_measure{0.0};
  std::size_t rule_count{0};
  std::size_t physical_rule_count{0};
  std::size_t skipped_physical_rule_count{0};
};

struct WetVolumeMeasureSelection {
  svmp::FE::Real wet_volume{0.0};
  std::string frame{"physical"};
};

CutVolumeMeasureSummary collectCutVolumeMeasures(
    const svmp::FE::assembly::IMeshAccess& mesh,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules);

WetVolumeMeasureSelection selectWetVolumeForDrift(
    const CutVolumeMeasureSummary& summary);

std::size_t writeWetVolumeFractionField(
    svmp::Mesh& mesh,
    const std::string& field_name,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules,
    const std::string& measure_field_name = std::string{});

struct WetVolumeDriftDiagnostic {
  svmp::FE::Real initial_wet_volume{0.0};
  svmp::FE::Real wet_volume_drift{0.0};
  svmp::FE::Real relative_wet_volume_drift{0.0};
};

WetVolumeDriftDiagnostic computeWetVolumeDrift(
    const std::string& key,
    svmp::FE::Real wet_volume,
    std::map<std::string, svmp::FE::Real>& initial_wet_volume_by_key);

} // namespace core
} // namespace application
