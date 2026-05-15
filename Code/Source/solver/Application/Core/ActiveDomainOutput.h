#pragma once

#include "FE/Geometry/CutQuadrature.h"
#include "Mesh/Mesh.h"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace application {
namespace core {

std::vector<double> collectWetVolumeFractions(
    std::size_t n_cells,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules);

std::size_t writeWetVolumeFractionField(
    svmp::Mesh& mesh,
    const std::string& field_name,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules);

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
