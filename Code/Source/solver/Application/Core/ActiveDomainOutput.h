#pragma once

#include "FE/Geometry/CutQuadrature.h"
#include "FE/LevelSet/LevelSetCutVolumeMeasures.h"
#include "Mesh/Mesh.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
class CutIntegrationContext;
class IMeshAccess;
}
}
}

namespace application {
namespace core {

// The owned cut-volume measures live in FE/LevelSet/LevelSetCutVolumeMeasures.h.
using svmp::FE::level_set::CutVolumeMeasureSummary;
using svmp::FE::level_set::collectCutVolumeMeasures;

std::vector<double> collectWetVolumeFractions(
    std::size_t n_cells,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules);

struct WetVolumeMeasureSelection {
  svmp::FE::Real wet_volume{0.0};
  std::string frame{"physical"};
};

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
